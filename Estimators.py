###Classes that define different off policy estimators for semi-synthetic experiments
import sys
import numpy
import scipy.sparse
import sklearn.model_selection
import sklearn.tree
import sklearn.linear_model
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import pdist
from sklearn.kernel_approximation import Nystroem, RBFSampler


class Estimator:
    # ranking_size: (int) Size of slate, l
    # logging_policy: (UniformPolicy) Logging policy, \mu
    # target_policy: (Policy) Target policy, \pi
    def __init__(self, ranking_size, logging_policy, target_policy):
        self.rankingSize = ranking_size
        self.name = None
        self.loggingPolicy = logging_policy
        self.targetPolicy = target_policy

        if target_policy.name is None or logging_policy.name is None:
            print("Estimator:init [ERR] Either target or logging policy is not initialized", flush=True)
            sys.exit(0)

        if target_policy.dataset.name != logging_policy.dataset.name:
            print("Estimator:init [ERR] Target and logging policy operate on different datasets", flush=True)
            sys.exit(0)

        ###All sub-classes of Estimator should supply a estimate method
        ###Requires: query, logged_ranking, logged_value,
        ###Returns: float indicating estimated value

        self.runningSum = 0
        self.runningMean = 0.0

    def updateRunningAverage(self, value):
        self.runningSum += 1
        delta = value - self.runningMean
        self.runningMean += delta / self.runningSum

    def reset(self):
        self.runningSum = 0
        self.runningMean = 0.0


class CME(Estimator):
    def __init__(self, ranking_size, logging_policy, target_policy, approx=False):
        Estimator.__init__(self, ranking_size, logging_policy, target_policy)
        if approx:
            self.name = 'CME_A'
        else:
            self.name = 'CME'
        self.numFeatures = self.loggingPolicy.dataset.features[0].shape[1]
        self.kernel = rbf_kernel
        self.kernel_param = 1.0
        self.reg_param = 0.001
        self.n_approx = 1
        self.p = 5000
        self.approx = approx

    def estimateAll(self, loggedData):
        numInstances = len(loggedData)
        targets = numpy.zeros(numInstances, order='C', dtype=numpy.float64)
        null_covariates = scipy.sparse.lil_matrix((numInstances, self.numFeatures * self.rankingSize),
                                                  dtype=numpy.float64)
        target_covariates = scipy.sparse.lil_matrix((numInstances, self.numFeatures * self.rankingSize),
                                                    dtype=numpy.float64)
        print("Starting to create covariates", flush=True)
        for j in range(numInstances):
            currentDatapoint = loggedData[j]

            targets[j] = currentDatapoint[2]

            currentQuery = currentDatapoint[0]
            currentRanking = currentDatapoint[1]
            newRanking = currentDatapoint[3]

            nullFeatures = self.loggingPolicy.dataset.features[currentQuery][currentRanking, :]
            nullFeatures.eliminate_zeros()

            targetFeatures = self.loggingPolicy.dataset.features[currentQuery][newRanking, :]
            targetFeatures.eliminate_zeros()

            null_covariates.data[j] = nullFeatures.data
            target_covariates.data[j] = targetFeatures.data
            nullIndices = nullFeatures.indices
            targetIndices = targetFeatures.indices
            for k in range(nullFeatures.shape[0]):
                nullIndices[nullFeatures.indptr[k]:nullFeatures.indptr[k + 1]] += k * self.numFeatures
                targetIndices[targetFeatures.indptr[k]:targetFeatures.indptr[k + 1]] += k * self.numFeatures

            null_covariates.rows[j] = nullIndices
            target_covariates.rows[j] = targetIndices

            if j % 1000 == 0:
                print(".", end='', flush=True)
            del currentDatapoint
            del nullFeatures
            del targetFeatures

        print("Converting covariates", flush=True)
        null_covariates = null_covariates.toarray()
        target_covariates = target_covariates.toarray()

        scaler = sklearn.preprocessing.MinMaxScaler()
        scaler.fit(null_covariates)

        s_null_covariates = scaler.transform(null_covariates)
        s_target_covariates = scaler.transform(target_covariates)
        print("Finished conversion", flush=True)

        print("Calculating heuristic kernel param", flush=True)

        random_indices = numpy.arange(0, s_null_covariates.shape[0])  # array of all indices
        numpy.random.shuffle(random_indices)  # shuffle the array

        sample_null_covar = s_null_covariates[random_indices[:2000]]  # get N samples without replacement

        sample_target_covar = s_target_covariates[random_indices[:2000]]  # get N samples without replacement

        recom_param = (0.5 * self.kernel_param) / numpy.median(
            pdist(numpy.vstack([sample_null_covar, sample_target_covar]), 'sqeuclidean'))

        print("Computing kernel matrix", flush=True)

        m = null_covariates.shape[0]
        n = target_covariates.shape[0]
        reg_params = self.reg_param / n

        if self.approx and m > self.p:
            p = self.p
            rets = []
            for i in range(self.n_approx):
                nystroem = Nystroem(gamma=recom_param, n_components=p)
                nystroem.fit(s_null_covariates)

                nullPhi = nystroem.transform(s_null_covariates)
                targetPhi = nystroem.transform(s_target_covariates)

                b = numpy.dot(targetPhi.T, numpy.repeat(1.0 / m, m, axis=0))
                A = nullPhi.T.dot(nullPhi) + numpy.diag(numpy.repeat(p * reg_params, p))
                beta_vec_approx = nullPhi.dot(scipy.sparse.linalg.cg(A, b, tol=1e-08, maxiter=5000)[0])

                ret = numpy.dot(beta_vec_approx, targets) / beta_vec_approx.sum()
                rets.append(ret)

            return numpy.mean(rets)
        else:
            nullRecomMatrix = self.kernel(s_null_covariates, s_null_covariates, recom_param)
            targetRecomMatrix = self.kernel(s_null_covariates, s_target_covariates, recom_param)

            b = numpy.dot(targetRecomMatrix, numpy.repeat(1.0 / m, m, axis=0))
            A = nullRecomMatrix + numpy.diag(numpy.repeat(n * reg_params, n))

            print("Finding beta_vec", flush=True)
            beta_vec, _ = scipy.sparse.linalg.cg(A, b, tol=1e-08, maxiter=5000)

            return numpy.dot(beta_vec, targets) / beta_vec.sum()


class OnPolicy(Estimator):
    def __init__(self, ranking_size, logging_policy, target_policy, metric):
        Estimator.__init__(self, ranking_size, logging_policy, target_policy)
        self.name = 'OnPolicy'
        self.metric = metric

        # This member is set on-demand by estimateAll(...)
        self.savedValues = None

    def estimateAll(self):
        if self.savedValues is not None:
            return

        self.savedValues = []
        numQueries = len(self.loggingPolicy.dataset.docsPerQuery)
        for i in range(numQueries):
            newRanking = self.targetPolicy.predict(i, self.rankingSize)
            self.savedValues.append(self.metric.computeMetric(i, newRanking))
            if i % 100 == 0:
                print(".", end="", flush=True)

        print("")
        print("OnPolicy:estimateAll [LOG] Precomputed estimates.", flush=True)

    def estimate(self, query, logged_ranking, new_ranking, logged_value):
        currentValue = None
        if self.savedValues is not None:
            currentValue = self.savedValues[query]
        else:
            currentValue = self.metric.computeMetric(query, new_ranking)

        self.updateRunningAverage(currentValue)
        return self.runningMean

    def reset(self):
        Estimator.reset(self)
        self.savedValues = None


class UniformIPS(Estimator):
    def __init__(self, ranking_size, logging_policy, target_policy):
        Estimator.__init__(self, ranking_size, logging_policy, target_policy)
        self.name = 'Unif-IPS'

    def estimate(self, query, logged_ranking, new_ranking, logged_value):
        exactMatch = numpy.absolute(new_ranking - logged_ranking).sum() == 0
        currentValue = 0.0
        if exactMatch:
            numAllowedDocs = self.loggingPolicy.dataset.docsPerQuery[query]
            validDocs = logged_ranking.size
            invPropensity = None
            if self.loggingPolicy.allowRepetitions:
                invPropensity = numpy.float_power(numAllowedDocs, validDocs)
            else:
                invPropensity = numpy.prod(range(numAllowedDocs + 1 - validDocs, numAllowedDocs + 1),
                                           dtype=numpy.float64)

            currentValue = logged_value * invPropensity

        self.updateRunningAverage(currentValue)
        return self.runningMean


class NonUniformIPS(Estimator):
    def __init__(self, ranking_size, logging_policy, target_policy):
        Estimator.__init__(self, ranking_size, logging_policy, target_policy)
        self.name = 'NonUnif-IPS'

    def estimate(self, query, logged_ranking, new_ranking, logged_value):
        exactMatch = numpy.absolute(new_ranking - logged_ranking).sum() == 0
        currentValue = 0.0
        if exactMatch:
            numAllowedDocs = self.loggingPolicy.dataset.docsPerQuery[query]
            underlyingRanking = self.loggingPolicy.policy.predict(query, -1)
            currentDistribution = self.loggingPolicy.multinomials[numAllowedDocs]

            numRankedDocs = logged_ranking.size
            invPropensity = 1.0
            denominator = 1.0
            for j in range(numRankedDocs):
                underlyingIndex = numpy.flatnonzero(underlyingRanking == logged_ranking[j])[0]
                invPropensity *= (denominator * 1.0 / currentDistribution[underlyingIndex])
                if not self.loggingPolicy.allowRepetitions:
                    denominator -= currentDistribution[underlyingIndex]

            currentValue = logged_value * invPropensity

        self.updateRunningAverage(currentValue)
        return self.runningMean


class UniformSNIPS(Estimator):
    def __init__(self, ranking_size, logging_policy, target_policy):
        Estimator.__init__(self, ranking_size, logging_policy, target_policy)
        self.name = 'Unif-IPS_SN'
        self.runningDenominatorMean = 0.0

    def estimate(self, query, logged_ranking, new_ranking, logged_value):
        exactMatch = numpy.absolute(new_ranking - logged_ranking).sum() == 0
        currentValue = 0.0
        if exactMatch:
            numAllowedDocs = self.loggingPolicy.dataset.docsPerQuery[query]
            validDocs = logged_ranking.size
            invPropensity = None
            if self.loggingPolicy.allowRepetitions:
                invPropensity = numpy.float_power(numAllowedDocs, validDocs)
            else:
                invPropensity = numpy.prod(range(numAllowedDocs + 1 - validDocs, numAllowedDocs + 1),
                                           dtype=numpy.float64)

            currentValue = logged_value * invPropensity

            self.updateRunningAverage(currentValue)
            denominatorDelta = invPropensity - self.runningDenominatorMean
            self.runningDenominatorMean += denominatorDelta / self.runningSum
        if self.runningDenominatorMean != 0.0:
            return 1.0 * self.runningMean / self.runningDenominatorMean
        else:
            return 0.0

    def reset(self):
        Estimator.reset(self)
        self.runningDenominatorMean = 0.0


class NonUniformSNIPS(Estimator):
    def __init__(self, ranking_size, logging_policy, target_policy, deterministic=True):
        Estimator.__init__(self, ranking_size, logging_policy, target_policy)
        self.name = 'NonUnif-IPS_SN'
        self.runningDenominatorMean = 0.0
        self.deterministic = deterministic

    def estimate(self, query, logged_ranking, new_ranking, logged_value):

        if self.deterministic:
            exactMatch = numpy.absolute(new_ranking - logged_ranking).sum() == 0
            currentValue = 0.0
            if exactMatch:
                numAllowedDocs = self.loggingPolicy.dataset.docsPerQuery[query]
                underlyingRanking = self.loggingPolicy.policy.predict(query, -1)
                currentDistribution = self.loggingPolicy.multinomials[numAllowedDocs]

                numRankedDocs = logged_ranking.size
                invPropensity = 1.0
                denominator = 1.0
                for j in range(numRankedDocs):
                    underlyingIndex = numpy.flatnonzero(underlyingRanking == logged_ranking[j])[0]
                    invPropensity *= (denominator * 1.0 / currentDistribution[underlyingIndex])
                    if not self.loggingPolicy.allowRepetitions:
                        denominator -= currentDistribution[underlyingIndex]

                currentValue = logged_value * invPropensity

                self.updateRunningAverage(currentValue)
                denominatorDelta = invPropensity - self.runningDenominatorMean
                self.runningDenominatorMean += denominatorDelta / self.runningSum
            if self.runningDenominatorMean != 0.0:
                return 1.0 * self.runningMean / self.runningDenominatorMean
            else:
                return 0.0
        else:
            numAllowedDocs = self.loggingPolicy.dataset.docsPerQuery[query]
            underlyingRanking = self.loggingPolicy.policy.predict(query, -1)
            currentDistribution = self.loggingPolicy.multinomials[numAllowedDocs]

            underlyingRankingTarget = self.targetPolicy.policy.predict(query, -1)
            currentDistributionTarget = self.targetPolicy.multinomials[numAllowedDocs]

            numRankedDocs = logged_ranking.size
            nullPropensity = 1.0
            nullDenominator = 1.0

            targetPropensity = 1.0
            targetDenominator = 1.0
            for j in range(numRankedDocs):
                underlyingIndex = numpy.flatnonzero(underlyingRanking == logged_ranking[j])[0]
                underlyingIndexTarget = numpy.flatnonzero(underlyingRankingTarget == logged_ranking[j])[0]
                nullPropensity *= (currentDistribution[underlyingIndex] / nullDenominator)
                targetPropensity *= (currentDistributionTarget[underlyingIndexTarget] / targetDenominator)
                if not self.loggingPolicy.allowRepetitions:
                    nullDenominator -= currentDistribution[underlyingIndex]
                    targetDenominator -= currentDistributionTarget[underlyingIndexTarget]

            currentValue = logged_value * targetPropensity / nullPropensity

            self.updateRunningAverage(currentValue)
            denominatorDelta = targetPropensity / nullPropensity - self.runningDenominatorMean
            self.runningDenominatorMean += denominatorDelta / self.runningSum

            if self.runningDenominatorMean != 0.0:
                return 1.0 * self.runningMean / self.runningDenominatorMean
            else:
                return 0.0

    def reset(self):
        Estimator.reset(self)
        self.runningDenominatorMean = 0.0


class UniformPI(Estimator):
    def __init__(self, ranking_size, logging_policy, target_policy):
        Estimator.__init__(self, ranking_size, logging_policy, target_policy)
        self.name = 'Unif-PI'

    def estimate(self, query, logged_ranking, new_ranking, logged_value):
        numAllowedDocs = self.loggingPolicy.dataset.docsPerQuery[query]
        validDocs = logged_ranking.size
        vectorDimension = validDocs * numAllowedDocs

        exploredMatrix = numpy.zeros((validDocs, numAllowedDocs), dtype=numpy.float64)
        newMatrix = numpy.zeros((validDocs, numAllowedDocs), dtype=numpy.float64)
        for j in range(validDocs):
            if self.loggingPolicy.dataset.mask is None:
                exploredMatrix[j, logged_ranking[j]] = 1
                newMatrix[j, new_ranking[j]] = 1
            else:
                logIndex = numpy.flatnonzero(self.loggingPolicy.dataset.mask[query] == logged_ranking[j])[0]
                newIndex = numpy.flatnonzero(self.loggingPolicy.dataset.mask[query] == new_ranking[j])[0]
                exploredMatrix[j, logIndex] = 1
                newMatrix[j, newIndex] = 1

        posRelVector = exploredMatrix.reshape(vectorDimension)
        newSlateVector = newMatrix.reshape(vectorDimension)

        estimatedPhi = numpy.dot(self.loggingPolicy.gammas[numAllowedDocs], posRelVector)
        invPropensity = numpy.dot(estimatedPhi, newSlateVector)

        currentValue = logged_value * invPropensity

        self.updateRunningAverage(currentValue)
        return self.runningMean


class NonUniformPI(Estimator):
    def __init__(self, ranking_size, logging_policy, target_policy):
        Estimator.__init__(self, ranking_size, logging_policy, target_policy)
        self.name = 'NonUnif-PI'

    def estimate(self, query, logged_ranking, new_ranking, logged_value):
        numAllowedDocs = self.loggingPolicy.dataset.docsPerQuery[query]
        underlyingRanking = self.loggingPolicy.policy.predict(query, -1)
        validDocs = logged_ranking.size
        vectorDimension = validDocs * numAllowedDocs

        exploredMatrix = numpy.zeros((validDocs, numAllowedDocs), dtype=numpy.float64)
        newMatrix = numpy.zeros((validDocs, numAllowedDocs), dtype=numpy.float64)
        for j in range(validDocs):
            logIndex = numpy.flatnonzero(underlyingRanking == logged_ranking[j])[0]
            newIndex = numpy.flatnonzero(underlyingRanking == new_ranking[j])[0]
            exploredMatrix[j, logIndex] = 1
            newMatrix[j, newIndex] = 1

        posRelVector = exploredMatrix.reshape(vectorDimension)
        newSlateVector = newMatrix.reshape(vectorDimension)

        estimatedPhi = numpy.dot(self.loggingPolicy.gammas[numAllowedDocs], posRelVector)
        invPropensity = numpy.dot(estimatedPhi, newSlateVector)

        currentValue = logged_value * invPropensity

        self.updateRunningAverage(currentValue)
        return self.runningMean


class UniformSNPI(Estimator):
    def __init__(self, ranking_size, logging_policy, target_policy):
        Estimator.__init__(self, ranking_size, logging_policy, target_policy)
        self.name = 'Unif-PI_SN'
        self.runningDenominatorMean = 0.0

    def estimate(self, query, logged_ranking, new_ranking, logged_value):
        numAllowedDocs = self.loggingPolicy.dataset.docsPerQuery[query]
        validDocs = logged_ranking.size
        vectorDimension = validDocs * numAllowedDocs

        exploredMatrix = numpy.zeros((validDocs, numAllowedDocs), dtype=numpy.float64)
        newMatrix = numpy.zeros((validDocs, numAllowedDocs), dtype=numpy.float64)
        for j in range(validDocs):
            if self.loggingPolicy.dataset.mask is None:
                exploredMatrix[j, logged_ranking[j]] = 1
                newMatrix[j, new_ranking[j]] = 1
            else:
                logIndex = numpy.flatnonzero(self.loggingPolicy.dataset.mask[query] == logged_ranking[j])[0]
                newIndex = numpy.flatnonzero(self.loggingPolicy.dataset.mask[query] == new_ranking[j])[0]
                exploredMatrix[j, logIndex] = 1
                newMatrix[j, newIndex] = 1

        posRelVector = exploredMatrix.reshape(vectorDimension)
        newSlateVector = newMatrix.reshape(vectorDimension)

        estimatedPhi = numpy.dot(self.loggingPolicy.gammas[numAllowedDocs], posRelVector)
        invPropensity = numpy.dot(estimatedPhi, newSlateVector)
        currentValue = logged_value * invPropensity

        self.updateRunningAverage(currentValue)

        denominatorDelta = invPropensity - self.runningDenominatorMean
        self.runningDenominatorMean += denominatorDelta / self.runningSum
        if self.runningDenominatorMean != 0.0:
            return 1.0 * self.runningMean / self.runningDenominatorMean
        else:
            return 0.0

    def reset(self):
        Estimator.reset(self)
        self.runningDenominatorMean = 0.0


class NonUniformSNPI(Estimator):
    def __init__(self, ranking_size, logging_policy, target_policy):
        Estimator.__init__(self, ranking_size, logging_policy, target_policy)
        self.name = 'NonUnif-PI_SN'
        self.runningDenominatorMean = 0.0

    def estimate(self, query, logged_ranking, new_ranking, logged_value):
        numAllowedDocs = self.loggingPolicy.dataset.docsPerQuery[query]
        underlyingRanking = self.loggingPolicy.policy.predict(query, -1)

        validDocs = logged_ranking.size
        vectorDimension = validDocs * numAllowedDocs

        exploredMatrix = numpy.zeros((validDocs, numAllowedDocs), dtype=numpy.float64)
        newMatrix = numpy.zeros((validDocs, numAllowedDocs), dtype=numpy.float64)
        for j in range(validDocs):
            logIndex = numpy.flatnonzero(underlyingRanking == logged_ranking[j])[0]
            newIndex = numpy.flatnonzero(underlyingRanking == new_ranking[j])[0]
            exploredMatrix[j, logIndex] = 1
            newMatrix[j, newIndex] = 1

        posRelVector = exploredMatrix.reshape(vectorDimension)
        newSlateVector = newMatrix.reshape(vectorDimension)

        estimatedPhi = numpy.dot(self.loggingPolicy.gammas[numAllowedDocs], posRelVector)
        invPropensity = numpy.dot(estimatedPhi, newSlateVector)
        currentValue = logged_value * invPropensity

        self.updateRunningAverage(currentValue)

        denominatorDelta = invPropensity - self.runningDenominatorMean
        self.runningDenominatorMean += denominatorDelta / self.runningSum
        if self.runningDenominatorMean != 0.0:
            return 1.0 * self.runningMean / self.runningDenominatorMean
        else:
            return 0.0

    def reset(self):
        Estimator.reset(self)
        self.runningDenominatorMean = 0.0


class Direct(Estimator):
    def __init__(self, ranking_size, logging_policy, target_policy, estimator_type):
        Estimator.__init__(self, ranking_size, logging_policy, target_policy)
        self.name = 'Direct_' + estimator_type
        self.estimatorType = estimator_type
        self.numFeatures = self.loggingPolicy.dataset.features[0].shape[1]
        self.hyperParams = {'alpha': (numpy.logspace(-2, 1, num=4, base=10)).tolist()}
        self.treeDepths = {'max_depth': list(range(3, 15, 3))}

        if self.estimatorType == 'tree':
            self.tree = None
        else:
            self.policyParams = None

        # This member is set on-demand by estimateAll(...)
        self.savedValues = None

    def estimateAll(self, metric=None):
        if self.savedValues is not None:
            return

        self.savedValues = []
        numQueries = len(self.loggingPolicy.dataset.docsPerQuery)
        for query in range(numQueries):
            newRanking = self.targetPolicy.predict(query, self.rankingSize)
            allFeatures = self.loggingPolicy.dataset.features[query][newRanking, :]

            if newRanking.size < self.rankingSize:
                emptyPad = scipy.sparse.csr_matrix((self.rankingSize - newRanking.size, self.numFeatures),
                                                   dtype=numpy.float64)
                allFeatures = scipy.sparse.vstack((allFeatures, emptyPad), format="csr", dtype=numpy.float64)

            allFeatures = allFeatures.toarray()
            nRows, nCols = allFeatures.shape
            size = nRows * nCols
            currentFeatures = numpy.reshape(allFeatures, (1, size))

            currentValue = None
            if self.estimatorType == 'tree':
                currentValue = self.tree.predict(currentFeatures)[0]
            else:
                currentValue = numpy.dot(currentFeatures, self.policyParams)[0]

            low = None
            high = None
            if metric is not None:
                low = metric.getMin(newRanking.size)
                high = metric.getMax(newRanking.size)

            if low is not None:
                currentValue = max(currentValue, low)
            if high is not None:
                currentValue = min(currentValue, high)

            if currentValue > 1.0 or currentValue < 0.0:
                print("Direct:estimateAll [LOG] estimate %0.3f " % (currentValue), flush=True)

            del allFeatures
            del currentFeatures

            self.savedValues.append(currentValue)

            if query % 100 == 0:
                print(".", end="", flush=True)

        print("")
        print("Direct:estimateAll [LOG] Precomputed estimates.", flush=True)

    def train(self, logged_data):
        numInstances = len(logged_data)
        targets = numpy.zeros(numInstances, order='C', dtype=numpy.float64)
        covariates = scipy.sparse.lil_matrix((numInstances, self.numFeatures * self.rankingSize), dtype=numpy.float64)
        print("Starting to create covariates", flush=True)
        for j in range(numInstances):
            currentDatapoint = logged_data[j]
            targets[j] = currentDatapoint[2]

            currentQuery = currentDatapoint[0]
            currentRanking = currentDatapoint[1]
            allFeatures = self.loggingPolicy.dataset.features[currentQuery][currentRanking, :]
            allFeatures.eliminate_zeros()

            covariates.data[j] = allFeatures.data
            newIndices = allFeatures.indices
            for k in range(allFeatures.shape[0]):
                newIndices[allFeatures.indptr[k]:allFeatures.indptr[k + 1]] += k * self.numFeatures

            covariates.rows[j] = newIndices

            if j % 1000 == 0:
                print(".", end='', flush=True)
            del currentDatapoint
            del allFeatures

        print("Converting covariates", flush=True)
        covariates = covariates.tocsr()
        print("Finished conversion", flush=True)

        if self.estimatorType == 'tree':
            treeCV = sklearn.model_selection.GridSearchCV(sklearn.tree.DecisionTreeRegressor(criterion="mse",
                                                                                             splitter="random",
                                                                                             min_samples_split=4,
                                                                                             min_samples_leaf=4,
                                                                                             presort=False),
                                                          param_grid=self.treeDepths,
                                                          scoring=None, fit_params=None, n_jobs=3,
                                                          iid=True, cv=3, refit=True, verbose=0, pre_dispatch=1,
                                                          error_score='raise', return_train_score=False)
            treeCV.fit(covariates, targets)
            self.tree = treeCV.best_estimator_
            print("DirectEstimator:train [INFO] Done. Best depth",
                  treeCV.best_params_['max_depth'], flush=True)
        elif self.estimatorType == 'lasso':
            lassoCV = sklearn.model_selection.GridSearchCV(sklearn.linear_model.Lasso(fit_intercept=False,
                                                                                      normalize=False, precompute=False,
                                                                                      copy_X=False,
                                                                                      max_iter=30000, tol=1e-4,
                                                                                      warm_start=False, positive=False,
                                                                                      random_state=None,
                                                                                      selection='random'),
                                                           param_grid=self.hyperParams,
                                                           scoring=None, fit_params=None, n_jobs=3,
                                                           iid=True, cv=3, refit=True, verbose=0, pre_dispatch=1,
                                                           error_score='raise', return_train_score=False)
            lassoCV.fit(covariates, targets)
            self.policyParams = lassoCV.best_estimator_.coef_
            print("DirectEstimator:train [INFO] Done. CVAlpha", lassoCV.best_params_['alpha'], flush=True)
        elif self.estimatorType == 'ridge':
            ridgeCV = sklearn.model_selection.GridSearchCV(sklearn.linear_model.Ridge(fit_intercept=False,
                                                                                      normalize=False, copy_X=False,
                                                                                      max_iter=30000, tol=1e-4,
                                                                                      solver='sag',
                                                                                      random_state=None),
                                                           param_grid=self.hyperParams,
                                                           scoring=None, fit_params=None, n_jobs=3,
                                                           iid=True, cv=3, refit=True, verbose=0, pre_dispatch=1,
                                                           error_score='raise', return_train_score=False)
            ridgeCV.fit(covariates, targets)
            self.policyParams = ridgeCV.best_estimator_.coef_
            print("DirectEstimator:train [INFO] Done. CVAlpha", ridgeCV.best_params_['alpha'], flush=True)
        else:
            print("DirectEstimator:train [ERR] %s not supported." % self.modelType, flush=True)
            sys.exit(0)

    def estimate(self, query, logged_ranking, new_ranking, logged_value):
        currentValue = None
        if self.savedValues is not None:
            currentValue = self.savedValues[query]
        else:
            allFeatures = self.loggingPolicy.dataset.features[query][new_ranking, :]

            if new_ranking.size < self.rankingSize:
                emptyPad = scipy.sparse.csr_matrix((self.rankingSize - new_ranking.size, self.numFeatures),
                                                   dtype=numpy.float64)
                allFeatures = scipy.sparse.vstack((allFeatures, emptyPad), format="csr", dtype=numpy.float64)

            allFeatures = allFeatures.toarray()
            nRows, nCols = allFeatures.shape
            size = nRows * nCols
            currentFeatures = numpy.reshape(allFeatures, (1, size))

            if self.estimatorType == 'tree':
                currentValue = self.tree.predict(currentFeatures)[0]
            else:
                currentValue = numpy.dot(currentFeatures, self.policyParams)[0]

            del allFeatures
            del currentFeatures

        self.updateRunningAverage(currentValue)
        return self.runningMean

    def reset(self):
        Estimator.reset(self)
        self.savedValues = None
        if self.estimatorType == 'tree':
            self.tree = None
        else:
            self.policyParams = None


class DoublyRobust(Direct):
    def __init__(self, ranking_size, logging_policy, target_policy, estimator_type, deterministic=True):
        super().__init__(ranking_size, logging_policy, target_policy, estimator_type)
        self.deterministic = deterministic

    def estimate(self, query, logged_ranking, new_ranking, logged_value):
        # DirectPrediction
        targetPred = self.savedValues[query]

        nullFeatures = self.loggingPolicy.dataset.features[query][logged_ranking, :]

        if new_ranking.size < self.rankingSize:
            emptyPad = scipy.sparse.csr_matrix((self.rankingSize - new_ranking.size, self.numFeatures),
                                               dtype=numpy.float64)
            nullFeatures = scipy.sparse.vstack((nullFeatures, emptyPad), format="csr", dtype=numpy.float64)

        nullFeatures = nullFeatures.toarray()
        nRows, nCols = nullFeatures.shape
        size = nRows * nCols
        currentFeatures = numpy.reshape(nullFeatures, (1, size))

        if self.estimatorType == 'tree':
            nullPred = self.tree.predict(currentFeatures)[0]
        else:
            nullPred = numpy.dot(currentFeatures, self.policyParams)[0]

        del nullFeatures
        del currentFeatures

        if self.deterministic:
            exactMatch = numpy.absolute(new_ranking - logged_ranking).sum() == 0

            ipsPred = 0.0
            if exactMatch:
                numAllowedDocs = self.loggingPolicy.dataset.docsPerQuery[query]
                underlyingRanking = self.loggingPolicy.policy.predict(query, -1)
                currentDistribution = self.loggingPolicy.multinomials[numAllowedDocs]

                numRankedDocs = logged_ranking.size
                invPropensity = 1.0
                denominator = 1.0
                for j in range(numRankedDocs):
                    underlyingIndex = numpy.flatnonzero(underlyingRanking == logged_ranking[j])[0]
                    invPropensity *= (denominator * 1.0 / currentDistribution[underlyingIndex])
                    if not self.loggingPolicy.allowRepetitions:
                        denominator -= currentDistribution[underlyingIndex]

                ipsPred = (logged_value - nullPred) * invPropensity
        else:
            # IPS Prediction
            numAllowedDocs = self.loggingPolicy.dataset.docsPerQuery[query]
            underlyingRanking = self.loggingPolicy.policy.predict(query, -1)
            currentDistribution = self.loggingPolicy.multinomials[numAllowedDocs]

            underlyingRankingTarget = self.targetPolicy.policy.predict(query, -1)
            currentDistributionTarget = self.targetPolicy.multinomials[numAllowedDocs]

            numRankedDocs = logged_ranking.size
            nullPropensity = 1.0
            nullDenominator = 1.0

            targetPropensity = 1.0
            targetDenominator = 1.0
            for j in range(numRankedDocs):
                underlyingIndex = numpy.flatnonzero(underlyingRanking == logged_ranking[j])[0]
                underlyingIndexTarget = numpy.flatnonzero(underlyingRankingTarget == logged_ranking[j])[0]
                nullPropensity *= (currentDistribution[underlyingIndex] / nullDenominator)
                targetPropensity *= (currentDistributionTarget[underlyingIndexTarget] / targetDenominator)
                if not self.loggingPolicy.allowRepetitions:
                    nullDenominator -= currentDistribution[underlyingIndex]
                    targetDenominator -= currentDistributionTarget[underlyingIndexTarget]

            ipsPred = (logged_value - nullPred) * targetPropensity / nullPropensity

        self.updateRunningAverage(targetPred + ipsPred)

        return self.runningMean

import numpy as np
from scipy import stats


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


def random_subset(n,n_data):
    # data array indexes
    all_idxs = np.arange(n_data)
    # shuffles the indexes
    np.random.shuffle(all_idxs)
    # the first n are returned
    idxs1 = all_idxs[:n]
    return idxs1


def get_error(test_points, maybeAngle):
    diffAngle = []
    for count, angle1 in enumerate(test_points):
        # Angles are converted to vectors
        vTest = [np.cos(angle1),np.sin(angle1)]
        vMaybe = [np.cos(maybeAngle),np.sin(maybeAngle)]
        # Difference is calculated using dot product
        escProd = np.dot(vTest,vMaybe)
        # limits the escProd to the range [-1,1]
        escProd = clamp(escProd, -1, 1)
        diffAngle.append(np.arccos(escProd))
    return np.array(diffAngle)


def ransac_thetas(data, n, k, t, d):
    """
    Fit model to data using the RANSAC algorithm
    Given:
        data - a set of observed data points
        n - the minimum number of data values required to compute an angle
        k - the maximum number of iterations allowed in the algorithm
        t - a threshold value for determining when a data point fits a model
        d - the number of close data values required to assert that a model fits well to data
    Return:
        bestfit          - angle that best fit the data (or nil if no good model is found)
        best_inlier_idxs - indexes of inlier data (or nil if no good model is found)
    """
    if type(data) is not np.array:
        data = np.array(data)
    it = 0
    bestfit = None
    besterr = np.inf
    best_inlier_idxs = None
    outlier_idxs = None
    while it < k:
        all_idxs = np.arange(data.shape[0])
        # generates a random subset
        maybe_idxs = random_subset(n,data.shape[0])
        maybeinliers = data[maybe_idxs]
        # generates a model out of selected inliers
        maybeAngle = stats.circmean(maybeinliers, high=np.pi, low=-np.pi)
        # error from all data to generated model
        test_err = get_error(data, maybeAngle)
        # select indices of rows with accepted points
        inl_idxs = all_idxs[test_err < t]
        inliers = data[inl_idxs]
        # if the number of inliers is above threshold
        if len(inliers) > d:
            bettermodel = maybeAngle
            better_errs = test_err
            thiserr = np.mean(better_errs)
            # if found error is the best so far, the model is updated
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = inl_idxs
        it += 1
    # returns best model and inliers' indexes
    else:
        if all_idxs is not None and best_inlier_idxs is not None:
            outlier_idxs = set(all_idxs).difference(best_inlier_idxs)
        return bestfit, best_inlier_idxs, outlier_idxs

# region problem statement
'''
    (50 pts) Using scipy and numpy:

    Re-work problem 1 from Exam 1, but now replace the Simpson integration method with quad from scipy.integration,
    and use fsolve in place of the Secant method. I’ve provided a copy of my solution to X1SP24_1.py as a reference
    along with my numericalMethods.py file.
'''
#Used ChatGPT to successfully help me implement both quad and fsolve
# endregion

# region imports
import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve


# endregion

# region function definitions

def lognormal_PDF(D, mu, sigma):
    """
    Computes the log-normal probability density function (PDF).

    :param D: Rock diameter
    :param mu: Mean of ln(D)
    :param sigma: Standard deviation of ln(D)
    :return: Log-normal probability density value
    """
    if D <= 0:
        return 0
    return (1 / (D * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(D) - mu) ** 2) / (2 * sigma ** 2))
# endregion

def compute_CDF(D, mu, sigma, D_Min):
    """
    Computes the cumulative probability of `D` using numerical integration (quad).

    :param D: Upper bound for probability computation
    :param mu: Mean of ln(D)
    :param sigma: Standard deviation of ln(D)
    :param D_Min: Lower integration bound (minimum sieve size)
    :return: Cumulative probability up to `D`
    """
    result, _ = quad(lambda x: lognormal_PDF(x, mu, sigma), D_Min, D)
    return result
# endregion

def tln_PDF(D, mu, sigma, F_DMin, F_DMax):
    """
    Computes the truncated log-normal probability density function (PDF).

    - Uses `quad` to normalize the probability between `D_Min` and `D_Max`.

    :param D: Rock diameter
    :param mu: Mean of ln(D)
    :param sigma: Standard deviation of ln(D)
    :param F_DMin: CDF value at D_Min
    :param F_DMax: CDF value at D_Max
    :return: Normalized probability density function value
    """
    return lognormal_PDF(D, mu, sigma) / (F_DMax - F_DMin)
# endregion

def F_tlnpdf(D, mu, sigma, D_Min, D_Max, F_DMax, F_DMin):
    """
    Computes the CDF of the truncated log-normal distribution using numerical integration (`quad`).

    :param D: Upper bound for probability computation
    :param mu: Mean of ln(D)
    :param sigma: Standard deviation of ln(D)
    :param D_Min: Minimum rock diameter
    :param D_Max: Maximum rock diameter
    :param F_DMax: CDF value at D_Max
    :param F_DMin: CDF value at D_Min
    :return: Cumulative probability for given D
    """
    if D < D_Min or D > D_Max:
        return 0
    result, _ = quad(lambda x: tln_PDF(x, mu, sigma, F_DMin, F_DMax), D_Min, D)
    return result
# endregion

def makeSample(mu, sigma, D_Min, D_Max, F_DMax, F_DMin, N=100):
    """
    Generates a sample of rock diameters based on the truncated log-normal distribution.

    - Uses `fsolve` to determine diameters corresponding to random probabilities.
    - Finds inverse CDF values to generate the sample set.

    :param mu: Mean of ln(D)
    :param sigma: Standard deviation of ln(D)
    :param D_Min: Minimum rock diameter
    :param D_Max: Maximum rock diameter
    :param F_DMax: CDF value at D_Max
    :param F_DMin: CDF value at D_Min
    :param N: Number of samples (default = 100)
    :return: List of sampled rock diameters
    """
    probs = np.random.rand(N)  # Generate random probabilities
    initial_guess = (D_Min + D_Max) / 2  # Use midpoint as initial guess
    d_s = []

    for prob in probs:
        try:
            root = fsolve(lambda D: F_tlnpdf(D, mu, sigma, D_Min, D_Max, F_DMax, F_DMin) - prob,
                          initial_guess, xtol=1e-5)[0]  # Solve for D
            d_s.append(root)
        except RuntimeWarning:
            print(f"Warning: fsolve failed for prob = {prob}, using default D_Min.")
            d_s.append(D_Min)  # Use fallback value if `fsolve` fails

    return d_s
# endregion

def sampleStats(D):
    """
    Computes the mean and variance of a given sample.

    :param D: List of sample values
    :return: Tuple containing (mean, variance)
    """
    return np.mean(D), np.var(D, ddof=1)
# endregion

def main():
    """
    Simulates an industrial-scale gravel sieving process with user-defined parameters.

    - Prompts the user to change values for:
        - μ (mean of ln(D))
        - σ (standard deviation of ln(D))
        - D_Min (minimum sieve size)
        - D_Max (maximum sieve size)
        - N_samples (number of samples)
        - N_sampleSize (sample size per iteration)

    - Uses `quad` for integration and `fsolve` for root-finding.
    - Computes and prints sample statistics.

    :return: None
    """
    # Get user input with default values
    mu = float(input("Mean of ln(D) for the pre-sieved rocks? (default=0.693) where D is in inches: ") or 0.693)
    sigma = float(input("Standard deviation of ln(D) for the pre-sieved rocks? (default=1.00): ") or 1.00)
    D_Max = float(input("Large aperture size? (default=1.0): ") or 1.0)
    D_Min = float(input("Small aperture size (default=0.375): ") or 0.375)
    N_samples = int(input("How many samples? (default=11): ") or 11)
    N_sampleSize = int(input("How many items in each sample? (default=100): ") or 100)

    # Compute CDF values using numerical integration (`quad`)
    F_DMax = compute_CDF(D_Max, mu, sigma, D_Min)
    F_DMin = compute_CDF(D_Min, mu, sigma, D_Min)

    # Generate samples and compute statistics
    Samples = []
    Means = []
    for n in range(N_samples):
        sample = makeSample(mu, sigma, D_Min, D_Max, F_DMax, F_DMin, N=N_sampleSize)
        Samples.append(sample)
        sample_mean, sample_variance = sampleStats(sample)
        Means.append(sample_mean)
        print(f"Sample {n + 1}: Mean = {sample_mean:.3f}, Variance = {sample_variance:.3f}")

    # Compute mean and variance of sample means
    mean_of_means, variance_of_means = sampleStats(Means)
    print(f"\nMean of Sample Means: {mean_of_means:.3f}")
    print(f"Variance of Sample Means: {variance_of_means:.6f}")


# endregion

# region function calls
if __name__ == "__main__":
    main()
# endregion
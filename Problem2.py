# region problem statement
'''
    Using matplotlib pyplot:

    In 2024, I assigned a homework problem to illustrate the connection between the Gaussian Normal Probability
    Density function and the Cumulative Distribution Function (see graphs below and attached file). For problem (b)
    of this homework, create a similar graph, but for the Truncated Log-Normal distribution after soliciting from the
    user values for the pre-sieved log-normal distribution and the sieved truncated log-normal distribution.

    Your graph should look like the one below where I’ve set the upper limit of integration and the corresponding
    F(D) at D = D_Min + (D_Max - D_Min) * 0.75. I’ve included a copy of the program HW4a.py from Spring 2024.

    Your program should produce the grey filled area, the axis labels, and the annotations on the upper plot.
'''
# endregion

# region imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy.integrate import quad

# endregion

def main():
    '''
    Generates and visualizes the Truncated Log-Normal Distribution.
    Asks the user for input parameters and then plots the probability density function (PDF) and
    cumulative distribution function (CDF) with annotations.
    :params: mu, sigma, D_Min, D_Max
    '''
    # ASk for user inputs (use default values for convenience)
    mu = float(input("Mean of ln(D) for the pre-sieved rocks? (default=0.693) where D is in inches: ") or 0.693)
    sigma = float(input("Standard deviation of ln(D) for the pre-sieved rocks? (default=1.00): ") or 1.00)
    D_Max = float(input("Large aperture size? (default=1.0): ") or 1.0)
    D_Min = float(input("Small aperture size (default=0.375): ") or 0.375)

    # Define the truncated log-normal distribution
    def truncated_ln_pdf(x, mu, sigma, D_Min, D_Max):
        if x < D_Min or x > D_Max:
            return 0
        pdf = lognorm.pdf(x, sigma, scale=np.exp(mu))
        normalization = lognorm.cdf(D_Max, sigma, scale=np.exp(mu)) - lognorm.cdf(D_Min, sigma, scale=np.exp(mu))
        return pdf / normalization

    # Create x values for plotting and computes PDF and CDF values
    x = np.linspace(D_Min, D_Max, 500)
    pdf_values = np.array([truncated_ln_pdf(xi, mu, sigma, D_Min, D_Max) for xi in x])
    cdf_values = np.array([quad(lambda xi: truncated_ln_pdf(xi, mu, sigma, D_Min, D_Max), D_Min, xi)[0] for xi in x])

    # Compute the upper limit for integration within the truncated range
    D_Upper = D_Min + (D_Max - D_Min) * 0.75
    P_DUpper, _ = quad(lambda xi: truncated_ln_pdf(xi, mu, sigma, D_Min, D_Max), D_Min, D_Upper)

    # Create the figure and two subplots
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    # Upper plot (PDF with shaded region)
    axes[0].plot(x, pdf_values, color='blue', label='Truncated Log-Normal PDF')
    axes[0].fill_between(x, pdf_values, where=(x <= D_Upper), color="gray", alpha=0.5)  # Shaded region
    axes[0].set_ylabel("f(D)")

    # Annotate PDF with LaTeX equation
    axes[0].text(D_Min + 0.02, max(pdf_values) * 0.85,
                 r"$f(D) = \frac{1}{D \sigma \sqrt{2\pi}} e^{-\frac{1}{2} \left( \frac{\ln(D) - \mu}{\sigma} \right)^2}$",
                 fontsize=12)

    # Add cumulative probability annotation with an arrow up to d_upper
    axes[0].annotate(
        rf"$P(D<{D_Upper:.2f} | TLN({mu:.2f},{sigma:.2f},{D_Min:.3f},{D_Max:.3f}))={P_DUpper:.2f}$",
        xy=(D_Upper, truncated_ln_pdf(D_Upper, mu, sigma, D_Min, D_Max)),
        xytext=(D_Min + 0.05, max(pdf_values) * 0.5),  # Position Adjustment
        arrowprops=dict(arrowstyle='->'), fontsize=12)

    # Lower plot: CDF with marked probability point
    axes[1].plot(x, cdf_values, color='blue', label='Truncated Log-Normal CDF')
    axes[1].scatter([D_Upper], [P_DUpper], color="red", edgecolors="black", zorder=3)  # Red marker at (D_Upper, P_DUpper)
    axes[1].axhline(P_DUpper, color="black", linestyle="--")  # Horizontal line
    axes[1].axvline(D_Upper, color="black", linestyle="--")  # Vertical line
    axes[1].set_xlabel("D")
    axes[1].set_ylabel(r"$\Phi(D) = \int_{D_{min}}^{D} f(D) dD$")

    # Show the final plot
    plt.show()
#Plot was made with help from ChatGPT
# endregion

if __name__ == "__main__":
    main()


    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm, t, binom

    sample = np.random.normal(loc=0, scale=1, size=1000)
    mu = np.mean(sample)
    sigma = np.std(sample)

    x_normal = np.linspace(-3*sigma, 3*sigma, 1000)
    pdf_normal = norm.pdf(x_normal, loc=mu, scale=sigma)

    plt.figure(figsize=(10, 6))
    plt.plot(x_normal, pdf_normal, label="Distribution normale", color="blue")
    plt.title("Distribution normale entre [-3σ, 3σ]")
    plt.grid()
    plt.show()


    x_t = np.linspace(-4, 4, 1000)
    pdf_t = t.pdf(x_t, df=len(sample)-1)

    plt.figure(figsize=(10, 6))
    plt.plot(x_t, pdf_t, label="Distribution t", color="orange")
    plt.title("Distribution t entre [-4, 4]")
    plt.grid()
    plt.show()

    n, p = 10, 0.5
    k = np.arange(0, n+1)
    binom = binom.pmf(k, n, p)

    plt.figure(figsize=(10, 6))
    plt.bar(k, binom, label="Distribution binomiale", color="green")
    plt.title("Distribution binomiale (n=10, p=0.5)")
    plt.xlabel("succès")
    plt.ylabel("Probabilité")
    plt.grid()
    plt.show()

    z_val, t_val = 0.5, 0.5
    prob_z = norm.cdf(z_val, loc=mu, scale=sigma)
    prob_t = t.cdf(t_val, df=len(sample)-1)

    print(f"Probabilité cumulée pour Z=0.5 : {prob_z:.4f}")
    print(f"Probabilité cumulée pour t=0.5 : {prob_t:.4f}")

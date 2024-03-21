# Goals

1. comprendre, expliquer et retrouver le microstructure noise

- offer theoretical explanation (a bit like in the quantstack post...)

2. retrouver quelques effets statistiques classiques du FOREX

- /!\ always look at log(price)
- seasonality (intraday/intraweek/etc.)
- mean-reversion at short timescales, then trending, then mean-reversion, then trending
- $\mathbb{E}[|\Delta x|] = c (\Delta t)^{1/E}$ for all timescales with $E$ the drift exponent ("scaling law")
- seasonal heteroskedascity of volatility (volatility clusters) across various timescales --> ask Gourevitch the link with ARCH/GARCH models [autocorrelation of squared returns is high at timescales ~hour and then decreases and peaks again for ~24h=1day]
- price change distributions become increasingly leptokurtic with decreasing-time intervals (cf. "aggregation gaussianity")
- check aggregational gaussianity of log(exchange rate)
- "Spot rates or their natural logarithms generally follow a random walk process."
- returns described by non-normal stable distribution, or a student distribution --> thus variance is not a correct measure of risk
  in the FOREX market
- "The specific days are selected so
  as to obtain information for quiet as well as turbulent periods, a distinction
  often considered important in the literature"
- "It turns out that all the series have to be first differenced because the level
  of the exchange rate is clearly non-stationary. For the statistical analysis
  presented in sections 4 and 5, the data are moreover first converted to
  natural logarithms to get the rates of return over a specific interval."
- look in "non-normal stable laws" (vs. student law)
- look at mean and std of returns, but also skewness and kurtosis!
- distinguish calm/turbulent days for analysis --> some statistical properties will change, others won't!
- "Unlike low-frequency data, high-frequency data
  have extremely high negative first-order autocorrelation in their return"

3. comprendre et reprendre dans le mémoire (brièvement) les mécaniques du FOREX (i.e. qui sont les acteurs, où, quand, comment, etc.)

Interesting (cf. Ren Tech) : we are building a trading machine which exploits human bias on the markets

MICROSTRUCTURE NOISE + EPPS EFFECT

Question

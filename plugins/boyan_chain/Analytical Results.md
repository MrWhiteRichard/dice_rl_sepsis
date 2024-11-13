$$
S = \{ s_0, \dots, s_N \},
\quad
A = \{ a_0, a_1 \},
$$

$$
\begin{equation*}
	\pi(a_j \mid s_i)
	=
	\begin{cases}
		p, & j = 0, \\
		1 - p, & j = 1,
	\end{cases}
	\quad \text{for} \quad
	i = 1, \dots, N.
\end{equation*}
$$

# `N = 0`

$$
\begin{align*}
	d_{\geq 0}^\pi(s_0) = 1,
\end{align*}
$$

$$
\begin{align*}
	d^{\pi, \gamma}(s_0) = 1,
\end{align*}
$$

# `N = 1`

$$
\begin{align*}
	d_0^\pi(s_i) & = \frac{1}{2}, \\
	\\
	d_{\geq 1}^\pi(s_i)
	& =
	\begin{cases}
		1, & i = 0, \\
		0, & i > 0, \\
	\end{cases}
\end{align*}
$$

$$
\begin{align*}
	d^{\pi, \gamma}(s_0)
	& =
	(1 - \gamma) \cdot \left (
		\frac{1}{2} + \sum_{t \geq 1} \gamma^t
	\right )
	\\ & =
	\frac{1 - \gamma}{2} \cdot \left (
		1 + \frac{2 \gamma}{1 - \gamma}
	\right ),
	\\ \\
	d^{\pi, \gamma}(s_1)
	& =
	(1 - \gamma) \cdot \frac{1}{2}
	\\ & =
	\frac{1 - \gamma}{2},
\end{align*}
$$

# `N = 2`

$$
\begin{align*}
	d_0^\pi(s_i) & = \frac{1}{3}, \\
	\\
	d_1^\pi(s_0) & =
		d_0^\pi(s_0) + d_0^\pi(s_1) + d_0^\pi(s_2) \cdot \pi(a_1 \mid s_2) =
		\frac{3 - p}{3}, \\
	d_1^\pi(s_1) & =
		d_0^\pi(s_2) \cdot \pi(a_0 \mid s_2) =
		\frac{p}{3}, \\
	d_1^\pi(s_2) & = 0, \\
	\\
	d_{\geq 2}^\pi(s_i)
	& =
	\begin{cases}
		1, & i = 0, \\
		0, & i > 0, \\
	\end{cases}
\end{align*}
$$

$$
\begin{align*}
	d^{\pi, \gamma}(s_0)
	& =
	(1 - \gamma) \cdot \left (
		\frac{1}{3} + \frac{3 - p}{3} \cdot \gamma + \sum_{t \geq 2} \gamma^t
	\right )
	\\ & =
	\frac{1 - \gamma}{3} \cdot \left (
		1 + (3 - p) \cdot \gamma + \frac{3 \gamma^2}{1 - \gamma}
	\right )
	\\ \\
	d^{\pi, \gamma}(s_1)
	& =
	(1 - \gamma) \cdot \left (
		\frac{1}{3} + \frac{p}{3} \cdot \gamma
	\right )
	\\ & =
	\frac{1 - \gamma}{3} \cdot (1 + p \cdot \gamma)
	\\ \\
	d^{\pi, \gamma}(s_2)
	& =
	(1 - \gamma) \cdot \frac{1}{3}
	\\ & =
	\frac{1 - \gamma}{3}
\end{align*}
$$

# `N = 3`

$$
\begin{align*}
	d_0^\pi(s_i) & = \frac{1}{4}, \\
	\\
	d_1^\pi(s_0) & =
		d_0^\pi(s_0) + d_0^\pi(s_1) + d_0^\pi(s_2) \cdot \pi(a_1 \mid s_2) =
		\frac{3 - p}{4}, \\
	d_1^\pi(s_1) & =
		d_0^\pi(s_2) \cdot \pi(a_0 \mid s_2)
			+ d_0^\pi(s_3) \cdot \pi(a_0 \mid s_3) =
		\frac{1}{4}, \\
	d_1^\pi(s_2) & =
		d_0^\pi(s_3) \cdot \pi(a_0 \mid s_3) =
		\frac{p}{4}, \\
	d_1^\pi(s_3) & = 0 \\
	\\
	d_2^\pi(s_0) & =
		d_1^\pi(s_0) + d_1^\pi(s_1) + d_1^\pi(s_2) \cdot \pi(a_1 \mid s_2) =
		\frac{4 - p^2}{4}, \\
	d_2^\pi(s_1) & =
		d_1^\pi(s_2) \cdot \pi(a_0 \mid s_2) =
		\frac{p^2}{4}, \\
	d_2^\pi(s_2) & = 0, \\
	d_2^\pi(s_3) & = 0 \\
	\\
	d_{\geq 3}^\pi(s_i)
	& =
	\begin{cases}
		1, & i = 0, \\
		0, & i > 0, \\
	\end{cases}
\end{align*}
$$

$$
\begin{align*}
	d^{\pi, \gamma}(s_0)
	& =
	(1 - \gamma) \cdot \left (
		\frac{1}{4}
		+ \frac{3 - p}{4} \cdot \gamma
		+ \frac{4 - p^2}{4} \cdot \gamma^2
		+ \sum_{t \geq 3} \gamma^t
	\right )
	\\ & =
	\frac{1 - \gamma}{4} \cdot \left (
		1
		+ (3 - p) \cdot \gamma
		+ (4 - p^2) \cdot \gamma^2
		+ \frac{4 \gamma^3}{1 - \gamma}
	\right )
	\\ \\
	d^{\pi, \gamma}(s_1)
	& =
	(1 - \gamma) \cdot \left (
		\frac{1}{4}
		+ \frac{1}{4} \cdot \gamma
		+ \frac{p^2}{4} \cdot \gamma^2
	\right )
	\\ & =
	\frac{1 - \gamma}{4} \cdot (1 + \gamma + p^2 \cdot \gamma^2)
	\\ \\
	d^{\pi, \gamma}(s_2)
	& =
	(1 - \gamma) \cdot \left (
		\frac{1}{4}
		+ \frac{p}{4} \cdot \gamma
	\right )
	\\ & =
	\frac{1 - \gamma}{4} \cdot (1 + p \cdot \gamma)
	\\ \\
	d^{\pi, \gamma}(s_3)
	& =
	(1 - \gamma) \cdot \frac{1}{4}
	\\ & =
	\frac{1 - \gamma}{4}
\end{align*}
$$

# `N >= 4`

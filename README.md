# 3D Vector Fields
In this project, I explore the generation and visualization of 3D vector fields using the python wrapper for the visualization toolkit vtk.

## 3D Simulation

### Free-space simulation

Maxwell's equations:

$$
\begin{matrix}
\frac{\partial \tilde{\vec{D}}}{\partial t} & = & \frac{1}{\sqrt{\epsilon_0\mu_0}} \nabla \times \vec{H} \qquad (1) \\

\tilde{\vec{D}}(\omega) & = &  \epsilon_r^*(\omega) \cdot \tilde{\vec{E}}(\omega) \qquad (2)\\

\frac{\partial \tilde{\vec{H}}}{\partial t} & = & -\frac{1}{\sqrt{\epsilon_0\mu_0}} \nabla \times \tilde{\vec{E}} \qquad (3)
\end{matrix}
$$

The vectors with the $\ \tilde{}$ notation are normalized in the following way:

$$
\begin{matrix}
\tilde{\vec{E}} & = & \sqrt{\frac{\epsilon_0}{\mu_0}} \vec{E} \qquad (4)\\
\tilde{\vec{D}} & = & \frac{1}{\sqrt{\epsilon_0 \mu_0}} \vec{D} \qquad (5)
\end{matrix}
$$

and in the general case we are dealing with a lossy dielectric medium with:

$$
\epsilon_r^*(\omega) = \epsilon_r + \frac{\sigma}{j \omega \epsilon_0} \qquad (6)
$$



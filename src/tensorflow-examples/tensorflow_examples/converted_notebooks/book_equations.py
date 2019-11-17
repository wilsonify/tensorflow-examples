#!/usr/bin/env python
# coding: utf-8

# **Equations**
# 
# *This notebook lists all the equations in the book. If you decide to print them on a T-Shirt, I definitely want a copy! ;-)*
# 
# **Warning**: GitHub's notebook viewer does not render equations properly. You should either view this notebook within Jupyter itself or use [Jupyter's online viewer](http://nbviewer.jupyter.org/github/ageron/handson-ml/blob/master/book_equations.ipynb).

# # Chapter 1
# **Equation 1-1: A simple linear model**
# 
# $
# \text{life_satisfaction} = \theta_0 + \theta_1 \times \text{GDP_per_capita}
# $
# 
# 

# # Chapter 2
# **Equation 2-1: Root Mean Square Error (RMSE)**
# 
# $
# \text{RMSE}(\mathbf{X}, h) = \sqrt{\frac{1}{m}\sum\limits_{i=1}^{m}\left(h(\mathbf{x}^{(i)}) - y^{(i)}\right)^2}
# $
# 
# 
# **Notations (page 38):**
# 
# $
#   \mathbf{x}^{(1)} = \begin{pmatrix}
#   -118.29 \\
#   33.91 \\
#   1,416 \\
#   38,372
#   \end{pmatrix}
# $
# 
# 
# $
#   y^{(1)}=156,400
# $
# 
# 
# $
#   \mathbf{X} = \begin{pmatrix}
#   (\mathbf{x}^{(1)})^T \\
#   (\mathbf{x}^{(2)})^T\\
#   \vdots \\
#   (\mathbf{x}^{(1999)})^T \\
#   (\mathbf{x}^{(2000)})^T
#   \end{pmatrix} = \begin{pmatrix}
#   -118.29 & 33.91 & 1,416 & 38,372 \\
#   \vdots & \vdots & \vdots & \vdots \\
#   \end{pmatrix}
# $
# 
# 
# **Equation 2-2: Mean Absolute Error**
# 
# $
# \text{MAE}(\mathbf{X}, h) = \frac{1}{m}\sum\limits_{i=1}^{m}\left| h(\mathbf{x}^{(i)}) - y^{(i)} \right|
# $
# 
# **$\ell_k$ norms (page 39):**
# 
# $ \left\| \mathbf{v} \right\| _k = (\left| v_0 \right|^k + \left| v_1 \right|^k + \dots + \left| v_n \right|^k)^{\frac{1}{k}} $
# 

# # Chapter 3
# **Equation 3-1: Precision**
# 
# $
# \text{precision} = \cfrac{TP}{TP + FP}
# $
# 
# 
# **Equation 3-2: Recall**
# 
# $
# \text{recall} = \cfrac{TP}{TP + FN}
# $
# 
# 
# **Equation 3-3: $F_1$ score**
# 
# $
# F_1 = \cfrac{2}{\cfrac{1}{\text{precision}} + \cfrac{1}{\text{recall}}} = 2 \times \cfrac{\text{precision}\, \times \, \text{recall}}{\text{precision}\, + \, \text{recall}} = \cfrac{TP}{TP + \cfrac{FN + FP}{2}}
# $
# 
# 

# # Chapter 4
# **Equation 4-1: Linear Regression model prediction**
# 
# $
# \hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n
# $
# 
# 
# **Equation 4-2: Linear Regression model prediction (vectorized form)**
# 
# $
# \hat{y} = h_{\boldsymbol{\theta}}(\mathbf{x}) = \boldsymbol{\theta} \cdot \mathbf{x}
# $
# 
# 
# **Equation 4-3: MSE cost function for a Linear Regression model**
# 
# $
# \text{MSE}(\mathbf{X}, h_{\boldsymbol{\theta}}) = \dfrac{1}{m} \sum\limits_{i=1}^{m}{(\boldsymbol{\theta}^T \mathbf{x}^{(i)} - y^{(i)})^2}
# $
# 
# 
# **Equation 4-4: Normal Equation**
# 
# $
# \hat{\boldsymbol{\theta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
# $
# 
# 
# ** Partial derivatives notation (page 114):**
# 
# $\frac{\partial}{\partial \theta_j} \text{MSE}(\boldsymbol{\theta})$
# 
# 
# **Equation 4-5: Partial derivatives of the cost function**
# 
# $
# \dfrac{\partial}{\partial \theta_j} \text{MSE}(\boldsymbol{\theta}) = \dfrac{2}{m}\sum\limits_{i=1}^{m}(\boldsymbol{\theta}^T \mathbf{x}^{(i)} - y^{(i)})\, x_j^{(i)}
# $
# 
# 
# **Equation 4-6: Gradient vector of the cost function**
# 
# $
# \nabla_{\boldsymbol{\theta}}\, \text{MSE}(\boldsymbol{\theta}) =
# \begin{pmatrix}
#  \frac{\partial}{\partial \theta_0} \text{MSE}(\boldsymbol{\theta}) \\
#  \frac{\partial}{\partial \theta_1} \text{MSE}(\boldsymbol{\theta}) \\
#  \vdots \\
#  \frac{\partial}{\partial \theta_n} \text{MSE}(\boldsymbol{\theta})
# \end{pmatrix}
#  = \dfrac{2}{m} \mathbf{X}^T (\mathbf{X} \boldsymbol{\theta} - \mathbf{y})
# $
# 
# 
# **Equation 4-7: Gradient Descent step**
# 
# $
# \boldsymbol{\theta}^{(\text{next step})} = \boldsymbol{\theta} - \eta \nabla_{\boldsymbol{\theta}}\, \text{MSE}(\boldsymbol{\theta})
# $
# 
# 
# $ O(\frac{1}{\text{iterations}}) $
# 
# 
# $ \hat{y} = 0.56 x_1^2 + 0.93 x_1 + 1.78 $
# 
# 
# $ y = 0.5 x_1^2 + 1.0 x_1 + 2.0 + \text{Gaussian noise} $
# 
# 
# $ \dfrac{(n+d)!}{d!\,n!} $
# 
# 
# $ \alpha \sum_{i=1}^{n}{{\theta_i}^2}$
# 
# 
# **Equation 4-8: Ridge Regression cost function**
# 
# $
# J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + \alpha \dfrac{1}{2}\sum\limits_{i=1}^{n}{\theta_i}^2
# $
# 
# 
# **Equation 4-9: Ridge Regression closed-form solution**
# 
# $
# \hat{\boldsymbol{\theta}} = (\mathbf{X}^T \mathbf{X} + \alpha \mathbf{A})^{-1} \mathbf{X}^T \mathbf{y}
# $
# 
# 
# **Equation 4-10: Lasso Regression cost function**
# 
# $
# J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + \alpha \sum\limits_{i=1}^{n}\left| \theta_i \right|
# $
# 
# 
# **Equation 4-11: Lasso Regression subgradient vector**
# 
# $
# g(\boldsymbol{\theta}, J) = \nabla_{\boldsymbol{\theta}}\, \text{MSE}(\boldsymbol{\theta}) + \alpha
# \begin{pmatrix}
#   \operatorname{sign}(\theta_1) \\
#   \operatorname{sign}(\theta_2) \\
#   \vdots \\
#   \operatorname{sign}(\theta_n) \\
# \end{pmatrix} \quad \text{where } \operatorname{sign}(\theta_i) =
# \begin{cases}
# -1 & \text{if } \theta_i < 0 \\
# 0 & \text{if } \theta_i = 0 \\
# +1 & \text{if } \theta_i > 0
# \end{cases}
# $
# 
# 
# **Equation 4-12: Elastic Net cost function**
# 
# $
# J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + r \alpha \sum\limits_{i=1}^{n}\left| \theta_i \right| + \dfrac{1 - r}{2} \alpha \sum\limits_{i=1}^{n}{{\theta_i}^2}
# $
# 
# 
# **Equation 4-13: Logistic Regression model estimated probability (vectorized form)**
# 
# $
# \hat{p} = h_{\boldsymbol{\theta}}(\mathbf{x}) = \sigma(\boldsymbol{\theta}^T \mathbf{x})
# $
# 
# 
# **Equation 4-14: Logistic function**
# 
# $
# \sigma(t) = \dfrac{1}{1 + \exp(-t)}
# $
# 
# 
# **Equation 4-15: Logistic Regression model prediction**
# 
# $
# \hat{y} =
# \begin{cases}
#   0 & \text{if } \hat{p} < 0.5, \\
#   1 & \text{if } \hat{p} \geq 0.5.
# \end{cases}
# $
# 
# 
# **Equation 4-16: Cost function of a single training instance**
# 
# $
# c(\boldsymbol{\theta}) =
# \begin{cases}
#   -\log(\hat{p}) & \text{if } y = 1, \\
#   -\log(1 - \hat{p}) & \text{if } y = 0.
# \end{cases}
# $
# 
# 
# **Equation 4-17: Logistic Regression cost function (log loss)**
# 
# $
# J(\boldsymbol{\theta}) = -\dfrac{1}{m} \sum\limits_{i=1}^{m}{\left[ y^{(i)} log\left(\hat{p}^{(i)}\right) + (1 - y^{(i)}) log\left(1 - \hat{p}^{(i)}\right)\right]}
# $
# 
# 
# **Equation 4-18: Logistic cost function partial derivatives**
# 
# $
# \dfrac{\partial}{\partial \theta_j} \text{J}(\boldsymbol{\theta}) = \dfrac{1}{m}\sum\limits_{i=1}^{m}\left(\mathbf{\sigma(\boldsymbol{\theta}}^T \mathbf{x}^{(i)}) - y^{(i)}\right)\, x_j^{(i)}
# $
# 
# 
# **Equation 4-19: Softmax score for class k**
# 
# $
# s_k(\mathbf{x}) = ({\boldsymbol{\theta}^{(k)}})^T \mathbf{x}
# $
# 
# 
# **Equation 4-20: Softmax function**
# 
# $
# \hat{p}_k = \sigma\left(\mathbf{s}(\mathbf{x})\right)_k = \dfrac{\exp\left(s_k(\mathbf{x})\right)}{\sum\limits_{j=1}^{K}{\exp\left(s_j(\mathbf{x})\right)}}
# $
# 
# 
# **Equation 4-21: Softmax Regression classifier prediction**
# 
# $
# \hat{y} = \underset{k}{\operatorname{argmax}} \, \sigma\left(\mathbf{s}(\mathbf{x})\right)_k = \underset{k}{\operatorname{argmax}} \, s_k(\mathbf{x}) = \underset{k}{\operatorname{argmax}} \, \left( ({\boldsymbol{\theta}^{(k)}})^T \mathbf{x} \right)
# $
# 
# 
# **Equation 4-22: Cross entropy cost function**
# 
# $
# J(\boldsymbol{\Theta}) = - \dfrac{1}{m}\sum\limits_{i=1}^{m}\sum\limits_{k=1}^{K}{y_k^{(i)}\log\left(\hat{p}_k^{(i)}\right)}
# $
# 
# **Cross entropy between two discrete probability distributions $p$ and $q$ (page 141):**
# $ H(p, q) = -\sum\limits_{x}p(x) \log q(x) $
# 
# 
# **Equation 4-23: Cross entropy gradient vector for class _k_**
# 
# $
# \nabla_{\boldsymbol{\theta}^{(k)}} \, J(\boldsymbol{\Theta}) = \dfrac{1}{m} \sum\limits_{i=1}^{m}{ \left ( \hat{p}^{(i)}_k - y_k^{(i)} \right ) \mathbf{x}^{(i)}}
# $
# 

# # Chapter 5
# **Equation 5-1: Gaussian RBF**
# 
# $
# {\displaystyle \phi_{\gamma}(\mathbf{x}, \boldsymbol{\ell})} = {\displaystyle \exp({\displaystyle -\gamma \left\| \mathbf{x} - \boldsymbol{\ell} \right\|^2})}
# $
# 
# 
# **Equation 5-2: Linear SVM classifier prediction**
# 
# $
# \hat{y} = \begin{cases}
#  0 & \text{if } \mathbf{w}^T \mathbf{x} + b < 0, \\
#  1 & \text{if } \mathbf{w}^T \mathbf{x} + b \geq 0
# \end{cases}
# $
# 
# 
# **Equation 5-3: Hard margin linear SVM classifier objective**
# 
# $
# \begin{split}
# &\underset{\mathbf{w}, b}{\operatorname{minimize}}\quad{\frac{1}{2}\mathbf{w}^T \mathbf{w}} \\
# &\text{subject to} \quad t^{(i)}(\mathbf{w}^T \mathbf{x}^{(i)} + b) \ge 1 \quad \text{for } i = 1, 2, \dots, m
# \end{split}
# $
# 
# 
# **Equation 5-4: Soft margin linear SVM classifier objective**
# 
# $
# \begin{split}
# &\underset{\mathbf{w}, b, \mathbf{\zeta}}{\operatorname{minimize}}\quad{\dfrac{1}{2}\mathbf{w}^T \mathbf{w} + C \sum\limits_{i=1}^m{\zeta^{(i)}}}\\
# &\text{subject to} \quad t^{(i)}(\mathbf{w}^T \mathbf{x}^{(i)} + b) \ge 1 - \zeta^{(i)} \quad \text{and} \quad \zeta^{(i)} \ge 0 \quad \text{for } i = 1, 2, \dots, m
# \end{split}
# $
# 
# 
# **Equation 5-5: Quadratic Programming problem**
# 
# $
# \begin{split}
# \underset{\mathbf{p}}{\text{Minimize}} \quad & \dfrac{1}{2} \mathbf{p}^T \mathbf{H} \mathbf{p} \quad + \quad \mathbf{f}^T \mathbf{p}  \\
# \text{subject to} \quad & \mathbf{A} \mathbf{p} \le \mathbf{b} \\
# \text{where } &
# \begin{cases}
#   \mathbf{p} & \text{ is an }n_p\text{-dimensional vector (} n_p = \text{number of parameters),}\\
#   \mathbf{H} & \text{ is an }n_p \times n_p \text{ matrix,}\\
#   \mathbf{f} & \text{ is an }n_p\text{-dimensional vector,}\\
#   \mathbf{A} & \text{ is an } n_c \times n_p \text{ matrix (}n_c = \text{number of constraints),}\\
#   \mathbf{b} & \text{ is an }n_c\text{-dimensional vector.}
# \end{cases}
# \end{split}
# $
# 
# 
# **Equation 5-6: Dual form of the linear SVM objective**
# 
# $
# \begin{split}
# \underset{\mathbf{\alpha}}{\operatorname{minimize}}
# \dfrac{1}{2}\sum\limits_{i=1}^{m}{
#   \sum\limits_{j=1}^{m}{
#   \alpha^{(i)} \alpha^{(j)} t^{(i)} t^{(j)} {\mathbf{x}^{(i)}}^T \mathbf{x}^{(j)}
#   }
# } \quad - \quad \sum\limits_{i=1}^{m}{\alpha^{(i)}}\\
# \text{subject to}\quad \alpha^{(i)} \ge 0 \quad \text{for }i = 1, 2, \dots, m
# \end{split}
# $
# 
# 
# **Equation 5-7: From the dual solution to the primal solution**
# 
# $
# \begin{split}
# &\hat{\mathbf{w}} = \sum_{i=1}^{m}{\hat{\alpha}}^{(i)}t^{(i)}\mathbf{x}^{(i)}\\
# &\hat{b} = \dfrac{1}{n_s}\sum\limits_{\scriptstyle i=1 \atop {\scriptstyle {\hat{\alpha}}^{(i)} > 0}}^{m}{\left(t^{(i)} - ({\hat{\mathbf{w}}}^T \mathbf{x}^{(i)})\right)}
# \end{split}
# $
# 
# 
# **Equation 5-8: Second-degree polynomial mapping**
# 
# $
# \phi\left(\mathbf{x}\right) = \phi\left( \begin{pmatrix}
#   x_1 \\
#   x_2
# \end{pmatrix} \right) = \begin{pmatrix}
#   {x_1}^2 \\
#   \sqrt{2} \, x_1 x_2 \\
#   {x_2}^2
# \end{pmatrix}
# $
# 
# 
# **Equation 5-9: Kernel trick for a 2^nd^-degree polynomial mapping**
# 
# $
# \begin{split}
# \phi(\mathbf{a})^T \phi(\mathbf{b}) & \quad = \begin{pmatrix}
#   {a_1}^2 \\
#   \sqrt{2} \, a_1 a_2 \\
#   {a_2}^2
#   \end{pmatrix}^T \begin{pmatrix}
#   {b_1}^2 \\
#   \sqrt{2} \, b_1 b_2 \\
#   {b_2}^2
# \end{pmatrix} = {a_1}^2 {b_1}^2 + 2 a_1 b_1 a_2 b_2 + {a_2}^2 {b_2}^2 \\
#  & \quad = \left( a_1 b_1 + a_2 b_2 \right)^2 = \left( \begin{pmatrix}
#   a_1 \\
#   a_2
# \end{pmatrix}^T \begin{pmatrix}
#     b_1 \\
#     b_2
#   \end{pmatrix} \right)^2 = (\mathbf{a}^T \mathbf{b})^2
# \end{split}
# $
# 
# **In the text about the kernel trick (page 162):**
# [...], then you can replace this dot product of transformed vectors simply by $ ({\mathbf{x}^{(i)}}^T  \mathbf{x}^{(j)})^2 $
# 
# 
# **Equation 5-10: Common kernels**
# 
# $
# \begin{split}
# \text{Linear:} & \quad K(\mathbf{a}, \mathbf{b}) = \mathbf{a}^T \mathbf{b} \\
# \text{Polynomial:} & \quad K(\mathbf{a}, \mathbf{b}) = \left(\gamma \mathbf{a}^T \mathbf{b} + r \right)^d \\
# \text{Gaussian RBF:} & \quad K(\mathbf{a}, \mathbf{b}) = \exp({\displaystyle -\gamma \left\| \mathbf{a} - \mathbf{b} \right\|^2}) \\
# \text{Sigmoid:} & \quad K(\mathbf{a}, \mathbf{b}) = \tanh\left(\gamma \mathbf{a}^T \mathbf{b} + r\right)
# \end{split}
# $
# 
# **Equation 5-11: Making predictions with a kernelized SVM**
# 
# $
# \begin{split}
# h_{\hat{\mathbf{w}}, \hat{b}}\left(\phi(\mathbf{x}^{(n)})\right) & = \,\hat{\mathbf{w}}^T \phi(\mathbf{x}^{(n)}) + \hat{b} = \left(\sum_{i=1}^{m}{\hat{\alpha}}^{(i)}t^{(i)}\phi(\mathbf{x}^{(i)})\right)^T \phi(\mathbf{x}^{(n)}) + \hat{b}\\
#  & = \, \sum_{i=1}^{m}{\hat{\alpha}}^{(i)}t^{(i)}\left(\phi(\mathbf{x}^{(i)})^T \phi(\mathbf{x}^{(n)})\right)  + \hat{b}\\
#  & = \sum\limits_{\scriptstyle i=1 \atop {\scriptstyle {\hat{\alpha}}^{(i)} > 0}}^{m}{\hat{\alpha}}^{(i)}t^{(i)} K(\mathbf{x}^{(i)}, \mathbf{x}^{(n)}) + \hat{b}
# \end{split}
# $
# 
# 
# **Equation 5-12: Computing the bias term using the kernel trick**
# 
# $
# \begin{split}
# \hat{b} & = \dfrac{1}{n_s}\sum\limits_{\scriptstyle i=1 \atop {\scriptstyle {\hat{\alpha}}^{(i)} > 0}}^{m}{\left(t^{(i)} - {\hat{\mathbf{w}}}^T \phi(\mathbf{x}^{(i)})\right)} = \dfrac{1}{n_s}\sum\limits_{\scriptstyle i=1 \atop {\scriptstyle {\hat{\alpha}}^{(i)} > 0}}^{m}{\left(t^{(i)} - {
#  \left(\sum_{j=1}^{m}{\hat{\alpha}}^{(j)}t^{(j)}\phi(\mathbf{x}^{(j)})\right)
#  }^T \phi(\mathbf{x}^{(i)})\right)}\\
#  & = \dfrac{1}{n_s}\sum\limits_{\scriptstyle i=1 \atop {\scriptstyle {\hat{\alpha}}^{(i)} > 0}}^{m}{\left(t^{(i)} -
# \sum\limits_{\scriptstyle j=1 \atop {\scriptstyle {\hat{\alpha}}^{(j)} > 0}}^{m}{
#   {\hat{\alpha}}^{(j)} t^{(j)} K(\mathbf{x}^{(i)},\mathbf{x}^{(j)})
# }
# \right)}
# \end{split}
# $
# 
# 
# **Equation 5-13: Linear SVM classifier cost function**
# 
# $
# J(\mathbf{w}, b) = \dfrac{1}{2} \mathbf{w}^T \mathbf{w} \quad + \quad C {\displaystyle \sum\limits_{i=1}^{m}max\left(0, t^{(i)} - (\mathbf{w}^T \mathbf{x}^{(i)} + b) \right)}
# $
# 
# 
# 

# # Chapter 6
# **Equation 6-1: Gini impurity**
# 
# $
# G_i = 1 - \sum\limits_{k=1}^{n}{{p_{i,k}}^2}
# $
# 
# 
# **Equation 6-2: CART cost function for classification**
# 
# $
# \begin{split}
# &J(k, t_k) = \dfrac{m_{\text{left}}}{m}G_\text{left} + \dfrac{m_{\text{right}}}{m}G_{\text{right}}\\
# &\text{where }\begin{cases}
# G_\text{left/right} \text{ measures the impurity of the left/right subset,}\\
# m_\text{left/right} \text{ is the number of instances in the left/right subset.}
# \end{cases}
# \end{split}
# $
# 
# **Entropy computation example (page 173):**
# 
# $ -\frac{49}{54}\log_2(\frac{49}{54}) - \frac{5}{54}\log_2(\frac{5}{54}) $
# 
# 
# **Equation 6-3: Entropy**
# 
# $
# H_i = -\sum\limits_{k=1 \atop p_{i,k} \ne 0}^{n}{{p_{i,k}}\log_2(p_{i,k})}
# $
# 
# 
# **Equation 6-4: CART cost function for regression**
# 
# $
# J(k, t_k) = \dfrac{m_{\text{left}}}{m}\text{MSE}_\text{left} + \dfrac{m_{\text{right}}}{m}\text{MSE}_{\text{right}} \quad
# \text{where }
# \begin{cases}
# \text{MSE}_{\text{node}} = \sum\limits_{\scriptstyle i \in \text{node}}(\hat{y}_{\text{node}} - y^{(i)})^2\\
# \hat{y}_\text{node} = \dfrac{1}{m_{\text{node}}}\sum\limits_{\scriptstyle i \in \text{node}}y^{(i)}
# \end{cases}
# $
# 

# # Chapter 7
# 
# **Equation 7-1: Weighted error rate of the $j^\text{th}$ predictor**
# 
# $
# r_j = \dfrac{\displaystyle \sum\limits_{\textstyle {i=1 \atop \hat{y}_j^{(i)} \ne y^{(i)}}}^{m}{w^{(i)}}}{\displaystyle \sum\limits_{i=1}^{m}{w^{(i)}}} \quad
# \text{where }\hat{y}_j^{(i)}\text{ is the }j^{\text{th}}\text{ predictor's prediction for the }i^{\text{th}}\text{ instance.}
# $
# 
# **Equation 7-2: Predictor weight**
# 
# $
# \begin{split}
# \alpha_j = \eta \log{\dfrac{1 - r_j}{r_j}}
# \end{split}
# $
# 
# 
# **Equation 7-3: Weight update rule**
# 
# $
# \begin{split}
# & \text{ for } i = 1, 2, \dots, m \\
# & w^{(i)} \leftarrow
# \begin{cases}
# w^{(i)} & \text{if }\hat{y_j}^{(i)} = y^{(i)}\\
# w^{(i)} \exp(\alpha_j) & \text{if }\hat{y_j}^{(i)} \ne y^{(i)}
# \end{cases}
# \end{split}
# $
# 
# **In the text page 194:**
# 
# Then all the instance weights are normalized (i.e., divided by $ \sum_{i=1}^{m}{w^{(i)}} $).
# 
# 
# **Equation 7-4: AdaBoost predictions**
# 
# $
# \hat{y}(\mathbf{x}) = \underset{k}{\operatorname{argmax}}{\sum\limits_{\scriptstyle j=1 \atop \scriptstyle \hat{y}_j(\mathbf{x}) = k}^{N}{\alpha_j}} \quad \text{where }N\text{ is the number of predictors.}
# $
# 
# 
# 

# # Chapter 8
# 
# **Equation 8-1: Principal components matrix**
# 
# $
# \mathbf{V}^T =
# \begin{pmatrix}
#   \mid & \mid & & \mid \\
#   \mathbf{c_1} & \mathbf{c_2} & \cdots & \mathbf{c_n} \\
#   \mid & \mid & & \mid
# \end{pmatrix}
# $
# 
# 
# **Equation 8-2: Projecting the training set down to _d_ dimensions**
# 
# $
# \mathbf{X}_{d\text{-proj}} = \mathbf{X} \mathbf{W}_d
# $
# 
# 
# **Equation 8-3: PCA inverse transformation, back to the original number of dimensions**
# 
# $
# \mathbf{X}_{\text{recovered}} = \mathbf{X}_{d\text{-proj}} {\mathbf{W}_d}^T
# $
# 
# 
# $ \sum_{j=1}^{m}{w_{i,j}\mathbf{x}^{(j)}} $
# 
# 
# **Equation 8-4: LLE step 1: linearly modeling local relationships**
# 
# $
# \begin{split}
# & \hat{\mathbf{W}} = \underset{\mathbf{W}}{\operatorname{argmin}}{\displaystyle \sum\limits_{i=1}^{m}} \left\|\mathbf{x}^{(i)} - \sum\limits_{j=1}^{m}{w_{i,j}}\mathbf{x}^{(j)}\right\|^2\\
# & \text{subject to }
# \begin{cases}
#   w_{i,j}=0 & \text{if }\mathbf{x}^{(j)} \text{ is not one of the }k\text{ c.n. of }\mathbf{x}^{(i)}\\
#   \sum\limits_{j=1}^{m}w_{i,j} = 1 & \text{for }i=1, 2, \dots, m
# \end{cases}
# \end{split}
# $
# 
# **In the text page 223:**
# 
# [...] then we want the squared distance between $\mathbf{z}^{(i)}$ and $ \sum_{j=1}^{m}{\hat{w}_{i,j}\mathbf{z}^{(j)}} $ to be as small as possible.
# 
# 
# **Equation 8-5: LLE step 2: reducing dimensionality while preserving relationships**
# 
# $
# \hat{\mathbf{Z}} = \underset{\mathbf{Z}}{\operatorname{argmin}}{\displaystyle \sum\limits_{i=1}^{m}} \left\|\mathbf{z}^{(i)} - \sum\limits_{j=1}^{m}{\hat{w}_{i,j}}\mathbf{z}^{(j)}\right\|^2
# $
# 

# # Chapter 9
# 
# **Equation 9-1: Rectified linear unit**
# 
# $
# h_{\mathbf{w}, b}(\mathbf{X}) = \max(\mathbf{X} \mathbf{w} + b, 0)
# $

# # Chapter 10
# 
# **Equation 10-1: Common step functions used in Perceptrons**
# 
# $
# \begin{split}
# \operatorname{heaviside}(z) =
# \begin{cases}
# 0 & \text{if }z < 0\\
# 1 & \text{if }z \ge 0
# \end{cases} & \quad\quad
# \operatorname{sgn}(z) =
# \begin{cases}
# -1 & \text{if }z < 0\\
# 0 & \text{if }z = 0\\
# +1 & \text{if }z > 0
# \end{cases}
# \end{split}
# $
# 
# 
# **Equation 10-2: Perceptron learning rule (weight update)**
# 
# $
# {w_{i,j}}^{(\text{next step})} = w_{i,j} + \eta (y_j - \hat{y}_j) x_i
# $
# 
# 
# **In the text page 266:**
# 
# It will be initialized randomly, using a truncated normal (Gaussian) distribution with a standard deviation of $ 2 / \sqrt{\text{n}_\text{inputs}} $.
# 

# # Chapter 11
# **Equation 11-1: Xavier initialization (when using the logistic activation function)**
# 
# $
# \begin{split}
# & \text{Normal distribution with mean 0 and standard deviation }
# \sigma = \sqrt{\dfrac{2}{n_\text{inputs} + n_\text{outputs}}}\\
# & \text{Or a uniform distribution between -r and +r, with }
# r = \sqrt{\dfrac{6}{n_\text{inputs} + n_\text{outputs}}}
# \end{split}
# $
# 
# **In the text page 278:**
# 
# When the number of input connections is roughly equal to the number of output
# connections, you get simpler equations (e.g., $ \sigma = 1 / \sqrt{n_\text{inputs}} $ or $ r = \sqrt{3} / \sqrt{n_\text{inputs}} $).
# 
# **Table 11-1: Initialization parameters for each type of activation function**
# 
# * Logistic uniform: $ r = \sqrt{\dfrac{6}{n_\text{inputs} + n_\text{outputs}}} $
# * Logistic normal: $ \sigma = \sqrt{\dfrac{2}{n_\text{inputs} + n_\text{outputs}}} $
# * Hyperbolic tangent uniform: $ r = 4 \sqrt{\dfrac{6}{n_\text{inputs} + n_\text{outputs}}} $
# * Hyperbolic tangent normal: $ \sigma = 4 \sqrt{\dfrac{2}{n_\text{inputs} + n_\text{outputs}}} $
# * ReLU (and its variants) uniform: $ r = \sqrt{2} \sqrt{\dfrac{6}{n_\text{inputs} + n_\text{outputs}}} $
# * ReLU (and its variants) normal: $ \sigma = \sqrt{2} \sqrt{\dfrac{2}{n_\text{inputs} + n_\text{outputs}}} $
# 
# **Equation 11-2: ELU activation function**
# 
# $
# \operatorname{ELU}_\alpha(z) =
# \begin{cases}
# \alpha(\exp(z) - 1) & \text{if } z < 0\\
# z & if z \ge 0
# \end{cases}
# $
# 
# 
# **Equation 11-3: Batch Normalization algorithm**
# 
# $
# \begin{split}
# 1.\quad & \mathbf{\mu}_B = \dfrac{1}{m_B}\sum\limits_{i=1}^{m_B}{\mathbf{x}^{(i)}}\\
# 2.\quad & {\mathbf{\sigma}_B}^2 = \dfrac{1}{m_B}\sum\limits_{i=1}^{m_B}{(\mathbf{x}^{(i)} - \mathbf{\mu}_B)^2}\\
# 3.\quad & \hat{\mathbf{x}}^{(i)} = \dfrac{\mathbf{x}^{(i)} - \mathbf{\mu}_B}{\sqrt{{\mathbf{\sigma}_B}^2 + \epsilon}}\\
# 4.\quad & \mathbf{z}^{(i)} = \gamma \hat{\mathbf{x}}^{(i)} + \beta
# \end{split}
# $
# 
# **In the text page 285:**
# 
# [...] given a new value $v$, the running average $v$ is updated through the equation:
# 
# $ \hat{v} \gets \hat{v} \times \text{momentum} + v \times (1 - \text{momentum}) $
# 
# **Equation 11-4: Momentum algorithm**
# 
# 1. $\mathbf{m} \gets \beta \mathbf{m} - \eta \nabla_\boldsymbol{\theta}J(\boldsymbol{\theta})$
# 2. $\boldsymbol{\theta} \gets \boldsymbol{\theta} + \mathbf{m}$
# 
# **In the text page 296:**
# 
# You can easily verify that if the gradient remains constant, the terminal velocity (i.e., the maximum size of the weight updates) is equal to that gradient multiplied by the learning rate η multiplied by $ \frac{1}{1 - \beta} $.
# 
# 
# **Equation 11-5: Nesterov Accelerated Gradient algorithm**
# 
# 1. $\mathbf{m} \gets \beta \mathbf{m} - \eta \nabla_\boldsymbol{\theta}J(\boldsymbol{\theta} + \beta \mathbf{m})$
# 2. $\boldsymbol{\theta} \gets \boldsymbol{\theta} + \mathbf{m}$
# 
# **Equation 11-6: AdaGrad algorithm**
# 
# 1. $\mathbf{s} \gets \mathbf{s} + \nabla_\boldsymbol{\theta}J(\boldsymbol{\theta}) \otimes \nabla_\boldsymbol{\theta}J(\boldsymbol{\theta})$
# 2. $\boldsymbol{\theta} \gets \boldsymbol{\theta} - \eta \, \nabla_\boldsymbol{\theta}J(\boldsymbol{\theta}) \oslash {\sqrt{\mathbf{s} + \epsilon}}$
# 
# **In the text page 298-299:**
# 
# This vectorized form is equivalent to computing $s_i \gets s_i + \left( \dfrac{\partial J(\boldsymbol{\theta})}{\partial \theta_i} \right)^2$ for each element $s_i$ of the vector $\mathbf{s}$.
# 
# **In the text page 299:**
# 
# This vectorized form is equivalent to computing $ \theta_i \gets \theta_i - \eta \, \dfrac{\partial J(\boldsymbol{\theta})}{\partial \theta_i} \dfrac{1}{\sqrt{s_i + \epsilon}} $ for all parameters $\theta_i$ (simultaneously).
# 
# 
# **Equation 11-7: RMSProp algorithm**
# 
# 1. $\mathbf{s} \gets \beta \mathbf{s} + (1 - \beta ) \nabla_\boldsymbol{\theta}J(\boldsymbol{\theta}) \otimes \nabla_\boldsymbol{\theta}J(\boldsymbol{\theta})$
# 2. $\boldsymbol{\theta} \gets \boldsymbol{\theta} - \eta \, \nabla_\boldsymbol{\theta}J(\boldsymbol{\theta}) \oslash {\sqrt{\mathbf{s} + \epsilon}}$
# 
# 
# **Equation 11-8: Adam algorithm**
# 
# 1. $\mathbf{m} \gets \beta_1 \mathbf{m} - (1 - \beta_1) \nabla_\boldsymbol{\theta}J(\boldsymbol{\theta})$
# 2. $\mathbf{s} \gets \beta_2 \mathbf{s} + (1 - \beta_2) \nabla_\boldsymbol{\theta}J(\boldsymbol{\theta}) \otimes \nabla_\boldsymbol{\theta}J(\boldsymbol{\theta})$
# 3. $\hat{\mathbf{m}} \gets \left(\dfrac{\mathbf{m}}{1 - {\beta_1}^T}\right)$
# 4. $\hat{\mathbf{s}} \gets \left(\dfrac{\mathbf{s}}{1 - {\beta_2}^T}\right)$
# 5. $\boldsymbol{\theta} \gets \boldsymbol{\theta} + \eta \, \hat{\mathbf{m}} \oslash {\sqrt{\hat{\mathbf{s}} + \epsilon}}$
# 
# **In the text page 309:**
# 
# We typically implement this constraint by computing $\left\| \mathbf{w} \right\|_2$ after each training step
# and clipping $\mathbf{w}$ if needed $ \left( \mathbf{w} \gets \mathbf{w} \dfrac{r}{\left\| \mathbf{w} \right\|_2} \right) $.
# 
# 
# 

# # Chapter 13
# 
# **Equation 13-1: Computing the output of a neuron in a convolutional layer**
# 
# $
# z_{i,j,k} = b_k + \sum\limits_{u = 0}^{f_h - 1} \, \, \sum\limits_{v = 0}^{f_w - 1} \, \, \sum\limits_{k' = 0}^{f_{n'} - 1} \, \, x_{i', j', k'} \times w_{u, v, k', k}
# \quad \text{with }
# \begin{cases}
# i' = i \times s_h + u \\
# j' = j \times s_w + v
# \end{cases}
# $
# 
# **Equation 13-2: Local response normalization**
# 
# $
# b_i = a_i  \left(k + \alpha \sum\limits_{j=j_\text{low}}^{j_\text{high}}{{a_j}^2} \right)^{-\beta} \quad \text{with }
# \begin{cases}
#   j_\text{high} = \min\left(i + \dfrac{r}{2}, f_n-1\right) \\
#   j_\text{low} = \max\left(0, i - \dfrac{r}{2}\right)
# \end{cases}
# $
# 
# 
# 

# # Chapter 14
# 
# **Equation 14-1: Output of a recurrent layer for a single instance**
# 
# $
# \mathbf{y}_{(t)} = \phi\left({\mathbf{W}_x}^T{\mathbf{x}_{(t)}} + {{\mathbf{W}_y}^T\mathbf{y}_{(t-1)}} + \mathbf{b} \right)
# $
# 
# 
# **Equation 14-2: Outputs of a layer of recurrent neurons for all instances in a mini-batch**
# 
# $
# \begin{split}
# \mathbf{Y}_{(t)} & = \phi\left(\mathbf{X}_{(t)} \mathbf{W}_{x} + \mathbf{Y}_{(t-1)} \mathbf{W}_{y} + \mathbf{b} \right) \\
# & = \phi\left(
# \left[\mathbf{X}_{(t)} \quad \mathbf{Y}_{(t-1)} \right]
#   \mathbf{W} + \mathbf{b} \right) \text{ with } \mathbf{W}=
# \left[ \begin{matrix}
#   \mathbf{W}_x\\
#   \mathbf{W}_y
# \end{matrix} \right]
# \end{split}
# $
# 
# **In the text page 391:**
# 
# Just like in regular backpropagation, there is a first forward pass through the unrolled network (represented by the dashed arrows); then the output sequence is evaluated using a cost function $ C(\mathbf{Y}_{(t_\text{min})}, \mathbf{Y}_{(t_\text{min}+1)}, \dots, \mathbf{Y}_{(t_\text{max})}) $ (where $t_\text{min}$ and $t_\text{max}$ are the first and last output time steps, not counting the ignored outputs)[...]
# 
# 
# **Equation 14-3: LSTM computations**
# 
# $
# \begin{split}
# \mathbf{i}_{(t)}&=\sigma({\mathbf{W}_{xi}}^T \mathbf{x}_{(t)} + {\mathbf{W}_{hi}}^T \mathbf{h}_{(t-1)} + \mathbf{b}_i)\\
# \mathbf{f}_{(t)}&=\sigma({\mathbf{W}_{xf}}^T \mathbf{x}_{(t)} + {\mathbf{W}_{hf}}^T \mathbf{h}_{(t-1)} + \mathbf{b}_f)\\
# \mathbf{o}_{(t)}&=\sigma({\mathbf{W}_{xo}}^T \mathbf{x}_{(t)} + {\mathbf{W}_{ho}}^T \mathbf{h}_{(t-1)} + \mathbf{b}_o)\\
# \mathbf{g}_{(t)}&=\operatorname{tanh}({\mathbf{W}_{xg}}^T \mathbf{x}_{(t)} + {\mathbf{W}_{hg}}^T \mathbf{h}_{(t-1)} + \mathbf{b}_g)\\
# \mathbf{c}_{(t)}&=\mathbf{f}_{(t)} \otimes \mathbf{c}_{(t-1)} \, + \, \mathbf{i}_{(t)} \otimes \mathbf{g}_{(t)}\\
# \mathbf{y}_{(t)}&=\mathbf{h}_{(t)} = \mathbf{o}_{(t)} \otimes \operatorname{tanh}(\mathbf{c}_{(t)})
# \end{split}
# $
# 
# 
# **Equation 14-4: GRU computations**
# 
# $
# \begin{split}
# \mathbf{z}_{(t)}&=\sigma({\mathbf{W}_{xz}}^T \mathbf{x}_{(t)} + {\mathbf{W}_{hz}}^T \mathbf{h}_{(t-1)}) \\
# \mathbf{r}_{(t)}&=\sigma({\mathbf{W}_{xr}}^T \mathbf{x}_{(t)} + {\mathbf{W}_{hr}}^T \mathbf{h}_{(t-1)}) \\
# \mathbf{g}_{(t)}&=\operatorname{tanh}\left({\mathbf{W}_{xg}}^T \mathbf{x}_{(t)} + {\mathbf{W}_{hg}}^T (\mathbf{r}_{(t)} \otimes \mathbf{h}_{(t-1)})\right) \\
# \mathbf{h}_{(t)}&=(1-\mathbf{z}_{(t)}) \otimes \mathbf{h}_{(t-1)} + \mathbf{z}_{(t)} \otimes \mathbf{g}_{(t)}
# \end{split}
# $
# 
# 
# 

# # Chapter 15
# 
# **Equation 15-1: Kullback–Leibler divergence**
# 
# $
# D_{\mathrm{KL}}(P\|Q) = \sum\limits_{i} P(i) \log \dfrac{P(i)}{Q(i)}
# $
# 
# 
# **Equation: KL divergence between the target sparsity _p_ and the actual sparsity _q_**
# 
# $
# D_{\mathrm{KL}}(p\|q) = p \, \log \dfrac{p}{q} + (1-p) \log \dfrac{1-p}{1-q}
# $
# 
# **In the text page 433:**
# 
# One common variant is to train the encoder to output $\gamma = \log\left(\sigma^2\right)$ rather than $\sigma$.
# Wherever we need $\sigma$ we can just compute $ \sigma = \exp\left(\dfrac{\gamma}{2}\right) $.
# 
# 
# 

# # Chapter 16
# 
# **Equation 16-1: Bellman Optimality Equation**
# 
# $
# V^*(s) = \underset{a}{\max}\sum\limits_{s'}{T(s, a, s') [R(s, a, s') + \gamma . V^*(s')]} \quad \text{for all }s
# $
# 
# **Equation 16-2: Value Iteration algorithm**
# 
# $
#   V_{k+1}(s) \gets \underset{a}{\max}\sum\limits_{s'}{T(s, a, s') [R(s, a, s') + \gamma . V_k(s')]} \quad \text{for all }s
# $
# 
# 
# **Equation 16-3: Q-Value Iteration algorithm**
# 
# $
#   Q_{k+1}(s, a) \gets \sum\limits_{s'}{T(s, a, s') [R(s, a, s') + \gamma . \underset{a'}{\max}\,{Q_k(s',a')}]} \quad \text{for all } (s,a)
# $
# 
# **In the text page 458:**
# 
# Once you have the optimal Q-Values, defining the optimal policy, noted $\pi^{*}(s)$, is trivial: when the agent is in state $s$, it should choose the action with the highest Q-Value for that state: $ \pi^{*}(s) = \underset{a}{\operatorname{argmax}} \, Q^*(s, a) $.
# 
# 
# **Equation 16-4: TD Learning algorithm**
# 
# $
# V_{k+1}(s) \gets (1-\alpha)V_k(s) + \alpha\left(r + \gamma . V_k(s')\right)
# $
# 
# 
# **Equation 16-5: Q-Learning algorithm**
# 
# $
# Q_{k+1}(s, a) \gets (1-\alpha)Q_k(s,a) + \alpha\left(r + \gamma . \underset{a'}{\max} \, Q_k(s', a')\right)
# $
# 
# 
# **Equation 16-6: Q-Learning using an exploration function**
# 
# $
# Q(s, a) \gets (1-\alpha)Q(s,a) + \alpha\left(r + \gamma \, \underset{a'}{\max}f(Q(s', a'), N(s', a'))\right)
# $
# 
# **Equation 16-7: Target Q-Value**
# 
# $
# y(s,a)=r+\gamma\,\max_{a'}\,Q_\boldsymbol\theta(s',a')
# $

# # Appendix A
# 
# Equations that appear in the text:
# 
# $
# \mathbf{H} =
# \begin{pmatrix}
# \mathbf{H'} & 0 & \cdots\\
# 0 & 0 & \\
# \vdots & & \ddots
# \end{pmatrix}
# $
# 
# 
# $
# \mathbf{A} =
# \begin{pmatrix}
# \mathbf{A'} & \mathbf{I}_m \\
# \mathbf{0} & -\mathbf{I}_m
# \end{pmatrix}
# $
# 
# 
# $ 1 - \frac{1}{5}^2 - \frac{4}{5}^2 $
# 
# 
# $ 1 - \frac{1}{2}^2 - \frac{1}{2}^2  $
# 
# 
# $ \frac{2}{5} \times $
# 
# 
# $ \frac{3}{5} \times 0 $

# # Appendix C

# Equations that appear in the text:
# 
# $ (\hat{x}, \hat{y}) $
# 
# 
# $ \hat{\alpha} $
# 
# 
# $ (\hat{x}, \hat{y}, \hat{\alpha}) $
# 
# 
# $
# \begin{cases}
# \frac{\partial}{\partial x}g(x, y, \alpha) = 2x - 3\alpha\\
# \frac{\partial}{\partial y}g(x, y, \alpha) = 2 - 2\alpha\\
# \frac{\partial}{\partial \alpha}g(x, y, \alpha) = -3x - 2y - 1\\
# \end{cases}
# $
# 
# 
# $ 2\hat{x} - 3\hat{\alpha} = 2 - 2\hat{\alpha} = -3\hat{x} - 2\hat{y} - 1 = 0 $
# 
# 
# $ \hat{x} = \frac{3}{2} $
# 
# 
# $ \hat{y} = -\frac{11}{4} $
# 
# 
# $ \hat{\alpha} = 1 $
# 
# 
# **Equation C-1: Generalized Lagrangian for the hard margin problem**
# 
# $
# \begin{split}
# \mathcal{L}(\mathbf{w}, b, \mathbf{\alpha}) = \frac{1}{2}\mathbf{w}^T \mathbf{w} - \sum\limits_{i=1}^{m}{\alpha^{(i)} \left(t^{(i)}(\mathbf{w}^T \mathbf{x}^{(i)} + b) - 1\right)} \\
# \text{with}\quad \alpha^{(i)} \ge 0 \quad \text{for }i = 1, 2, \dots, m
# \end{split}
# $
# 
# **More equations in the text:**
# 
# $ (\hat{\mathbf{w}}, \hat{b}, \hat{\mathbf{\alpha}}) $
# 
# 
# $ t^{(i)}(\hat{\mathbf{w}}^T \mathbf{x}^{(i)} + \hat{b}) \ge 1 \quad \text{for } i = 1, 2, \dots, m $
# 
# 
# $ {\hat{\alpha}}^{(i)} \ge 0 \quad \text{for } i = 1, 2, \dots, m $
# 
# 
# $ {\hat{\alpha}}^{(i)} = 0 $
# 
# 
# $ t^{(i)}((\hat{\mathbf{w}})^T \mathbf{x}^{(i)} + \hat{b}) = 1 $
# 
# 
# $ {\hat{\alpha}}^{(i)} = 0 $
# 
# 
# **Equation C-2: Partial derivatives of the generalized Lagrangian**
# 
# $
# \begin{split}
# \nabla_{\mathbf{w}}\mathcal{L}(\mathbf{w}, b, \mathbf{\alpha}) = \mathbf{w} - \sum\limits_{i=1}^{m}\alpha^{(i)}t^{(i)}\mathbf{x}^{(i)}\\
# \dfrac{\partial}{\partial b}\mathcal{L}(\mathbf{w}, b, \mathbf{\alpha}) = -\sum\limits_{i=1}^{m}\alpha^{(i)}t^{(i)}
# \end{split}
# $
# 
# 
# **Equation C-3: Properties of the stationary points**
# 
# $
# \begin{split}
# \hat{\mathbf{w}} = \sum_{i=1}^{m}{\hat{\alpha}}^{(i)}t^{(i)}\mathbf{x}^{(i)}\\
# \sum_{i=1}^{m}{\hat{\alpha}}^{(i)}t^{(i)} = 0
# \end{split}
# $
# 
# 
# **Equation C-4: Dual form of the SVM problem**
# 
# $
# \begin{split}
# \mathcal{L}(\hat{\mathbf{w}}, \hat{b}, \mathbf{\alpha}) = \dfrac{1}{2}\sum\limits_{i=1}^{m}{
#   \sum\limits_{j=1}^{m}{
#   \alpha^{(i)} \alpha^{(j)} t^{(i)} t^{(j)} {\mathbf{x}^{(i)}}^T \mathbf{x}^{(j)}
#   }
# } \quad - \quad \sum\limits_{i=1}^{m}{\alpha^{(i)}}\\
# \text{with}\quad \alpha^{(i)} \ge 0 \quad \text{for }i = 1, 2, \dots, m
# \end{split}
# $
# 
# **Some more equations in the text:**
# 
# $ \hat{\mathbf{\alpha}} $
# 
# 
# $ {\hat{\alpha}}^{(i)} \ge 0 $
# 
# 
# $ \hat{\mathbf{\alpha}} $
# 
# 
# $ \hat{\mathbf{w}} $
# 
# 
# $ \hat{b} $
# 
# 
# $ \hat{b} = t^{(k)} - {\hat{\mathbf{w}}}^T \mathbf{x}^{(k)} $
# 
# 
# **Equation C-5: Bias term estimation using the dual form**
# 
# $
# \hat{b} = \dfrac{1}{n_s}\sum\limits_{\scriptstyle i=1 \atop {\scriptstyle {\hat{\alpha}}^{(i)} > 0}}^{m}{\left[t^{(i)} - {\hat{\mathbf{w}}}^T \mathbf{x}^{(i)}\right]}
# $

# # Appendix D

# **Equation D-1: Partial derivatives of $f(x,y)$**
# 
# $
# \begin{split}
# \dfrac{\partial f}{\partial x} & = \dfrac{\partial(x^2y)}{\partial x} + \dfrac{\partial y}{\partial x} + \dfrac{\partial 2}{\partial x} = y \dfrac{\partial(x^2)}{\partial x} + 0 + 0 = 2xy \\
# \dfrac{\partial f}{\partial y} & = \dfrac{\partial(x^2y)}{\partial y} + \dfrac{\partial y}{\partial y} + \dfrac{\partial 2}{\partial y} = x^2 + 1 + 0 = x^2 + 1 \\
# \end{split}
# $
# 
# **In the text:**
# 
# $ \frac{\partial g}{\partial x} = 0 + (0 \times x + y \times 1) = y $
# 
# 
# $ \frac{\partial x}{\partial x} = 1 $
# 
# 
# $ \frac{\partial y}{\partial x} = 0 $
# 
# 
# $ \frac{\partial (u \times v)}{\partial x} = \frac{\partial v}{\partial x} \times u + \frac{\partial u}{\partial x} \times u  $
# 
# 
# $ \frac{\partial g}{\partial x} = 0 + (0 \times x + y \times 1)  $
# 
# 
# $ \frac{\partial g}{\partial x} = y $
# 
# 
# **Equation D-2: Derivative of a function _h_(_x_) at point _x_~0~**
# 
# $
# \begin{split}
# h'(x) & = \underset{\textstyle x \to x_0}{\lim}\dfrac{h(x) - h(x_0)}{x - x_0}\\
#       & = \underset{\textstyle \epsilon \to 0}{\lim}\dfrac{h(x_0 + \epsilon) - h(x_0)}{\epsilon}
# \end{split}
# $
# 
# 
# **Equation D-3: A few operations with dual numbers**
# 
# $
# \begin{split}
# &\lambda(a + b\epsilon) = \lambda a + \lambda b \epsilon\\
# &(a + b\epsilon) + (c + d\epsilon) = (a + c) + (b + d)\epsilon \\
# &(a + b\epsilon) \times (c + d\epsilon) = ac + (ad + bc)\epsilon + (bd)\epsilon^2 = ac + (ad + bc)\epsilon\\
# \end{split}
# $
# 
# **In the text:**
# 
# $ \frac{\partial f}{\partial x}(3, 4) $
# 
# 
# $ \frac{\partial f}{\partial y}(3, 4) $
# 
# 
# **Equation D-4: Chain rule**
# 
# $
# \dfrac{\partial f}{\partial x} = \dfrac{\partial f}{\partial n_i} \times \dfrac{\partial n_i}{\partial x}
# $
# 
# **In the text:**
# 
# $ \frac{\partial f}{\partial n_7} = 1 $
# 
# 
# $ \frac{\partial f}{\partial n_5} = \frac{\partial f}{\partial n_7} \times \frac{\partial n_7}{\partial n_5} $
# 
# 
# $ \frac{\partial f}{\partial n_7} = 1 $
# 
# 
# $ \frac{\partial n_7}{\partial n_5} $
# 
# 
# $ \frac{\partial n_7}{\partial n_5} = 1 $
# 
# 
# $ \frac{\partial f}{\partial n_5} = 1 \times 1 = 1 $
# 
# 
# $ \frac{\partial f}{\partial n_4} = \frac{\partial f}{\partial n_5} \times \frac{\partial n_5}{\partial n_4} $
# 
# 
# $ \frac{\partial n_5}{\partial n_4} = n_2 $
# 
# 
# $ \frac{\partial f}{\partial n_4} = 1 \times n_2 = 4 $
# 
# 
# $ \frac{\partial f}{\partial x} = 24 $
# 
# 
# $ \frac{\partial f}{\partial y} = 10 $

# # Appendix E

# **Equation E-1: Probability that the i^th^ neuron will output 1**
# 
# $
# p\left(s_i^{(\text{next step})} = 1\right) \, = \, \sigma\left(\frac{\textstyle \sum\limits_{j = 1}^N{w_{i,j}s_j + b_i}}{\textstyle T}\right)
# $
# 
# **In the text:**
# 
# $ \dot{\mathbf{x}} $
# 
# 
# $ \dot{\mathbf{h}} $
# 
# 
# **Equation E-2: Contrastive divergence weight update**
# 
# $
# w_{i,j}^{(\text{next step})} = w_{i,j} + \eta(\mathbf{x}\mathbf{h}^T - \dot{\mathbf{x}} \dot {\mathbf{h}}^T)
# $

# # Glossary
# 
# In the text:
# 
# $\ell _1$
# 
# 
# $\ell _2$
# 
# 
# $\ell _k$
# 
# 
# $ \chi^2 $
# 

# Just in case your eyes hurt after all these equations, let's finish with the single most beautiful equation in the world. No, it's not $E = mc²$, it's obviously Euler's identity:

# $e^{i\pi}+1=0$

# In[ ]:





---
layout: post
title:  "Representation of a Three-Dimensional Moving Scene"
date:   2019-01-03 09:00:13
categories: slam

---

# Representation of a Three-Dimensional Moving Scene

Reference: An Invitation to 3-D vision, Chapter 2. 

## CheatSheet

|         | Rotation | Transformation |
|:--------|:--------|:--------|
| vector form | $ w \in \mathbb{R}^3 $ | $ \xi = \begin{bmatrix} w & v \end{bmatrix}^T \in \mathbb{R}^6 $<br>: twist coordinates |
| exponential coordinates | $ \hat{w} \in so(3) \subset \mathbb{R}^{3 \times 3} $  $$ \widehat{w} \doteq  \begin{bmatrix} 0 & -w_3 & w_2 \\ w_3 & 0 & -w_1 \\ -w_2 & w_1 & 0 \end{bmatrix} $$ | $ \hat{\xi} \in se(3) \subset \mathbb{R}^{4 \times 4} $: twist  $$ \widehat{\xi} \doteq \begin{bmatrix} \widehat{w} & v \\ 0 & 0 \end{bmatrix} $$ |
| matrix form | $ R = e^{\hat{w}} \in SO(3) \subset \mathbb{R}^{3 \times 3} $ | $ T = e^{\hat{\xi}} \in SE(3) \subset \mathbb{R}^{4 \times 4} $ |
| exponential map | $$e^{\widehat{w}t} = I + \widehat{w}sin(t) + \widehat{w} ^2 (1-cos(t))$$ where $$ \begin{Vmatrix} w \end{Vmatrix}=1 $$ | $$ e^{\widehat{\xi}t}  = \begin{bmatrix} e^{\widehat{w}} & \left( I - e^{\widehat{w}} \right) \widehat{w} v + w \widehat{w}^T v \\ 0 & 1 \end{bmatrix} $$ |



## 2.1 Three-dimensional Euclidean space

A point in the Euclidean space:

$$
\mathbf{X} \doteq \begin{bmatrix} X_1 \\ X_2 \\ X_3 \end{bmatrix} \in \mathbb{R}^3,   \mathbb{E}^3
$$

A vector in the Euclidean space 

$$
v \doteq \mathbf{Y - X} \in \mathbb{R}^3
$$

Inner product of two vectors:

$$
\left\langle u,v \right\rangle \doteq u^Tv = u_1 v_1 + u_2 v_2 + u_3 v_3, \forall u,v \in \mathbb{R}^3
$$

Cross product of two vectors:

$$
u \times v \doteq \begin{bmatrix} u_2v_3 - u_3v_2 \\ u_3v_1 - u_1v_3 \\ u_1v_2 - u_2v_1 \end{bmatrix} \in \mathbb{R}^3
$$

For fixed $ u $, the cross product can be represented by a map from $ \mathbb{R}^3 $  to $ \mathbb{R}^3: u \mapsto u \times v $. We define this mapping matrix by

$$
\widehat{u} \doteq  \begin{bmatrix} 0 & -u_3 & u_2 \\ u_3 & 0 & -u_1 \\ -u_2 & u_1 & 0 \end{bmatrix} \in \mathbb{R}^{3 \times 3}
$$

such that $ u \times v = \widehat{u}v $. Note that $ \widehat{u} $ is a skew-symmetric matrix, i.e. $ \widehat{u}^T = -\widehat{u} $.   

The cross product defines a (one-to-one) map between a vector  u  and a $3 \times 3$ skew-symmetric matrix  $ \widehat{u} $. The map is defined by a *hat-operator*

$$
\wedge : \mathbb{R}^3 \rightarrow so(3); u \mapsto \widehat{u}.
$$


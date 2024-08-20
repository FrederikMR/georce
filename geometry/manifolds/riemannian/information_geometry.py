#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:51:29 2024

@author: fmry
"""

#%% Sources

#%% Modules

from geometry.setup import *

####################

from .manifold import RiemannianManifold

#%% Code

#https://arxiv.org/pdf/2304.14885
class FisherRaoGeometry(RiemannianManifold):
    def __init__(self,
                 distribution:str="Gaussian",
                 params:Dict = None,
                 )->None:
        
        distributions = ['Binomial',
                         'Poisson',
                         'Geometric',
                         'Negative Binomial',
                         'Categorial',
                         'Multinomial',
                         'Negative Multinomial',
                         'Exponential',
                         'Rayleigh',
                         'Erlang',
                         'Gaussian',
                         'Laplace',
                         'Generalised Gaussian',
                         'Logistic',
                         'Cauchy',
                         'Students t',
                         'Log-Gaussian',
                         'Inverse Gaussian',
                         'Gumbel',
                         'Frechet',
                         'Weibull',
                         'Pareto',
                         'Power Function',
                         'Wishart',
                         'Inverse Wishart',
                         ]
        
        if distribution in distributions:
            self.distribution = distribution
        else:
            raise ValueError(f"Distribution, {self.distribution}, not implemented!")
            
        self.params = params
        
        if "Wishart" in distribution:
            self.Dn = self.DupMat(params['n'])
            self.Dp = self.invDupMat(params['n'])
        
        g, self.dim = self.get_metric()

        super().__init__(G=g)
        
        return
    
    def __str__(self)->str:
        
        return f"Information Geometry for distribution {self.distribution}"
    
    def DupMat(self, N:int):
        
        def step_col(carry, i, j):
            
            D, A, u, i = carry
            A,u = 0.*A, 0.*u
            idx = j*N+i-((j+1)*j)//2
            
            A = A.at[i,i-j].set(1)
            A = A.at[i-j,i].set(1)
            u = u.at[idx].set(1)
            D += u.dot(A.reshape((1, -1), order="F"))
            i += 1
            
            return (D,A,u,i), None
            
        p = N*(N+1)//2
        A,D,u = jnp.zeros((N,N)), jnp.zeros((p,N*N)), jnp.zeros((p,1))    
        
        for j in range(N):
            D, _, _, _ = lax.scan(lambda carry, i: step_col(carry, i, j), init=(D,A,u,j), xs=jnp.arange(0,N-j,1))[0]
        
        return D.T

    def invDupMat(self, N:int):
        
        def step_col(carry, i, j, val):
            
            D, A, u, i = carry
            A,u = 0.*A, 0.*u
            idx = j*N+i-((j+1)*j)//2
            
            A = A.at[i,i-j].set(val)
            A = A.at[i-j,i].set(val)
            u = u.at[idx].set(1)
            D += u.dot(A.reshape((1, -1), order="F"))
            i += 1
            
            return (D,A,u,i), None
            
        p = N*(N+1)//2
        A,D,u = jnp.zeros((N,N)), jnp.zeros((p,N*N)), jnp.zeros((p,1))    
        
        D, _, _, _ = lax.scan(lambda carry, i: step_col(carry, i, 0, 1.0), init=(D,A,u,0), xs=jnp.arange(0,N,1))[0]
        for j in range(1,N):
            D, _, _, _ = lax.scan(lambda carry, i: step_col(carry, i, j, 0.5), init=(D,A,u,j), xs=jnp.arange(0,N-j,1))[0]
        
        return D
    
    def get_metric(self)->Tuple[Callable, int]:
        
        if self.distribution == 'Binomial':
            g, dim = self.G_binomial, 1
        elif self.distribution == 'Poisson':
            g, dim = self.G_poisson, 1
        elif self.distribution == 'Geometric':
            g, dim = self.G_geometric, 1
        elif self.distribution == 'Negative Binomial':
            g, dim = self.G_negative_binomial, 1
        elif self.distribution == 'Categorial':
            g, dim = self.G_categorial, self.params['n']-1
        elif self.distribution == 'Multinomial':
            g, dim = self.G_multinomial, self.params['n']-1
        elif self.distribution == 'Negative Multinomial':
            g, dim = self.G_negative_multinomial, self.params['n']-1
        elif self.distribution == 'Exponential':
            g, dim = self.G_exponential, 1
        elif self.distribution == 'Rayleigh':
            g, dim = self.G_rayleigh, 1
        elif self.distribution == 'Erlang':
            g, dim = self.G_erlang, 1
        elif self.distribution == 'Gaussian':
            g, dim = self.G_gaussian, 2
        elif self.distribution == 'Laplace':
            g, dim = self.G_laplace, 2
        elif self.distribution == 'Generalised Gaussian':
            g, dim = self.G_generalised_gaussian, 2
        elif self.distribution == 'Logistic':
            g, dim = self.G_logistic, 2
        elif self.distribution == 'Cauchy':
            g, dim = self.G_cauchy, 2
        elif self.distribution == 'Students t':
            g, dim = self.G_students_t, 2
        elif self.distribution == 'Log-Gaussian':
            g, dim = self.G_log_gaussian, 2
        elif self.distribution == 'Inverse Gaussian':
            g, dim = self.G_inverse_gaussian, 2
        elif self.distribution == 'Gumbel':
            g, dim = self.G_gumbel, 2
        elif self.distribution == 'Frechet':
            g, dim = self.G_frechet, 2
        elif self.distribution == 'Weibull':
            g, dim = self.G_weibull, 2
        elif self.distribution == 'Pareto':
            g, dim = self.G_pareto, 2
        elif self.distribution == 'Power Function':
            g, dim = self.G_power_function, 2
        elif self.distribution == 'Wishart':
            g, dim = self.G_wishart, self.params['n']**2
        elif self.distribution == 'Inverse Wishart':
            g, dim = self.G_inverse_wishart, self.params['n']**2
        else:
            raise ValueError(f"Distribution, {self.distribution}, is not defined")
        
        return g, dim
    
    def G_binomial(self, z:Array)->Array:
        
        theta = z
        n = self.params['n']
        
        return jnp.array([n/(theta*(1.0-theta))]).reshape(1,1)
    
    def G_poisson(self, z:Array)->Array:
        
        lam = z
        
        return jnp.array([1/lam]).reshape(1,1)
    
    def G_geometric(self, z:Array)->Array:
        
        theta = z
        
        return jnp.array([1/((1-theta)*(theta**2))]).reshape(1,1)
    
    def G_negative_binomial(self, z:Array)->Array:
        
        theta = z
        r = self.params['r']
        
        return jnp.array([r/((1-theta)*(theta**2))]).reshape(1,1)
    
    def G_categorial(self, z:Array)->Array:
        
        pi = z
        pn = 1.-jnp.sum(pi)
        
        return jnp.eye(len(pi))/pi+jnp.ones((len(pi),len(pi)))/pn
    
    def G_multinomial(self, z:Array)->Array:
        
        pi = self.params['pi']
        pn = 1.-jnp.sum(pi)
        
        return len(pi)*jnp.eye(len(pi))/pi+len(pn)*jnp.ones((len(pi),len(pi)))/pn
    
    def G_negative_multinomial(self, z:Array)->Array:
        
        pi = z
        xn = self.params['xn']
        pn = 1.-jnp.sum(pi)
        
        return xn*jnp.eye(len(pi))/(pi*pn)+xn*jnp.ones((len(pi),len(pi)))/(pn**2)
    
    def G_exponential(self, z:Array)->Array:
        
        lam = z
        
        return jnp.array([1/(lam**2)]).reshape(1,1)
    
    def G_rayleigh(self, z:Array)->Array:
        
        sigma2 = z**2
        
        return jnp.array([4./(sigma2)]).reshape(1,1)
    
    def G_erlang(self, z:Array)->Array:
        
        lam2 = z**2
        k = self.params['k']
        
        return jnp.array([k/(lam2)]).reshape(1,1)
    
    def G_gaussian(self, z:Array)->Array:
        
        mu, sigma2 = z[0], z[1]**2
        
        return jnp.array([[1.0/(sigma2), 0],
                          [0, 2.0/(sigma2)]
                          ])
    
    def G_laplace(self, z:Array)->Array:
        
        mu, sigma2 = z[0], z[1]**2
        
        return jnp.array([[1.0/(sigma2), 0],
                          [0, 1.0/(sigma2)]
                          ])
    
    def G_generalised_gaussian(self, z:Array)->Array:
        
        mu, sigma2 = z[0], z[1]**2
        beta = self.params['beta']
        
        gamma1 = jscipy.special.gamma(2.0-1.0/beta)
        gamma2 = jscipy.special.gamma(1.0+1.0/beta)
        
        return jnp.array([[beta*gamma1/((sigma2)*gamma2), 0],
                          [0, beta/((sigma2))]
                          ])
    
    def G_logistic(self, z:Array)->Array:
        
        mu, sigma2 = z[0], z[1]**2

        return jnp.array([[1.0/(3.0*sigma2), 0],
                          [0, (jnp.pi**2+3)/(9.0*sigma2)]
                          ])
    
    def G_cauchy(self, z:Array)->Array:
        
        mu, sigma2 = z[0], z[1]**2

        return jnp.array([[1.0/(2.0*sigma2), 0],
                          [0, 1.0/(2.0*sigma2)]
                          ])
    
    def G_students_t(self, z:Array)->Array:
        
        v = self.params['v']
        mu, sigma2 = z[0], z[1]**2

        return jnp.array([[(v+1.0)/((v+3)*sigma2), 0],
                          [0, (2.0*v)/((v+3.0)*sigma2)]
                          ])
    
    def G_log_gaussian(self, z:Array)->Array:
        
        mu, sigma2 = z[0], z[1]**2

        return jnp.array([[1.0/sigma2, 0],
                          [0, 2.0/sigma2]
                          ])
    
    def G_inverse_gaussian(self, z:Array)->Array:
        
        lam, mu = z[0], z[1]

        return jnp.array([[1.0/(2.0*lam**2), 0],
                          [0, lam/(mu**3)]
                          ])
    
    def G_gumbel(self, z:Array)->Array:
        
        mu, sigma2 = z[0], z[1]**2
        gamma = self.params['gamma']

        return jnp.array([[1.0/sigma2, (gamma-1.0)/sigma2],
                          [(gamma-1.0)/sigma2, ((gamma-1.0)**2+(jnp.pi**2)/6.0)/sigma2]
                          ])
    
    def G_frechet(self, z:Array)->Array:
        
        beta, lam = z[0], z[1]
        gamma = 0.57721566490153286060651209008240243104215933593992 #Eulers constant
        
        return jnp.array([[(lam**2)/beta**2, (1.0-gamma)/beta],
                          [(1.0-gamma)/beta, ((gamma-1.0)**2+(jnp.pi**2)/6.0)/(lam**2)]
                          ])
    
    def G_weibull(self, z:Array)->Array:
        
        beta, lam = z[0], z[1]
        
        return jnp.array([[(lam**2)/beta**2, (gamma-1.0)/beta],
                          [(gamma-1.0)/beta, ((gamma-1.0)**2+(jnp.pi**2)/6.0)/(lam**2)]
                          ])
    
    def G_pareto(self, z:Array)->Array:
        
        theta2, alpha2 = z[0]**2, z[1]**2
        
        return jnp.array([[1.0/theta2, 0.0],
                          [0.0, theta2/alpha2]
                          ])
    
    def G_power_function(self, z:Array)->Array:
        
        theta2, alpha2 = z[0]**2, z[1]**2
        
        return jnp.array([[1.0/theta2, 0.0],
                          [0.0, theta2/alpha2]
                          ])
    
    def G_wishart(self, z:Array)->Array:
        
        sigma = z.reshape(self.params['n'], self.params['n'])
        sigma_inv = jnp.linalg.inv(sigma)
        Dn = self.Dn
        
        return 0.5*len(sigma)*jnp.einsum('ik,ij,jl->kl', 
                                         Dn,
                                         jnp.kron(sigma_inv, sigma_inv),
                                         Dn
                                         )
    
    def G_inverse_wishart(self, z:Array)->Array:
        
        sigma = z.reshape(self.params['n'], self.params['n'])
        sigma_inv = jnp.linalg.inv(sigma)
        Dn = self.Dn
        
        return 0.5*len(sigma)*jnp.einsum('ik,ij,jl->kl', 
                                         Dn,
                                         jnp.kron(sigma_inv, sigma_inv),
                                         Dn
                                         )
    
    def pdf_exponential(self, x:Array, z:Array, *args)->Array:
        
        lam = z
        
        return lam*jnp.exp(-lam*x)
    
    def pdf_rayleigh(self, x:Array, z:Array, *args)->Array:
        
        sigma2 = z**2
        
        return jnp.exp(-(x**2)/(2*sigma2))*x/sigma2
    
    def pdf_erlang(self, x:Array, z:Array, *args)->Array:
        
        lam = z
        k = args[0]

        return (lam**k)*(x**(k-1))*jnp.exp(-lam*x)/jscipy.factorial(k-1)

    def pdf_gaussian(self, x:Array, z:Array, *args)->Array:
        
        mu, sigma2 = z[0], z[1]**2
        
        return jnp.exp(-((x-mu)**2)/(2*sigma2))/jnp.sqrt(2*jnp.pi*sigma2)
    
    def pdf_laplace(self, x:Array, z:Array, *args)->Array:
        
        mu, sigma = z[0], z[1]
        
        return jnp.exp(-((x-mu)**2)/sigma)/(2*sigma)
    
    def pdf_generalised_gaussian(self, x:Array, z:Array, *args)->Array:
        
        mu, sigma = z[0], z[1]
        beta = args[0]
        
        return beta*jnp.exp(-((x-mu)**beta)/sigma)/(2*sigma*jscipy.special.gamma(1.0/beta))
    
    def pdf_logistic(self, x:Array, z:Array, *args)->Array:
        
        mu, sigma = z[0], z[1]
        
        num = jnp.exp(-(x-mu)/sigma)
        den = sigma*((jnp.exp(-(x-mu)/sigma)+1)**2)
        
        return num/den
    
    def pdf_cauchy(self, x:Array, z:Array, *args)->Array:
        
        mu, sigma = z[0], z[1]
        
        num = sigma
        den = jnp.pi*(((x-mu)**2)+sigma**2)
        
        return num/den
    
    def pdf_students_t(self, x:Array, z:Array, *args)->Array:
        
        mu, sigma = z[0], z[1]
        v = args[0]
        
        term1 = 1.0+(((x-mu)/sigma)**2)/v
        term2 = jscipy.special.gamma((v+1)/2)/(sigma*jnp.sqrt(jnp.pi*v)*jscipy.special.gamma(v/2))
        
        return term2*(term1**(-(v+1)/2))
    
    def pdf_log_gaussian(self, x:Array, z:Array, *args)->Array:
        
        mu, sigma = z[0], z[1]
        
        return jnp.exp(-(jnp.log(x)-mu)/(2*sigma**2))/(sigma*x*jnp.sqrt(2*jnp.pi))
    
    def pdf_inverse_gaussian(self, x:Array, z:Array, *args)->Array:
        
        lam, mu = z[0], z[1]
        
        term1 = jnp.sqrt(lam/(2*jnp.pi*(x**3)))
        term2 = jnp.exp(-lam*((x-mu)**2)/(2*(mu**2)*x))
        
        return term1*term2
    
    def pdf_gumbel(self, x:Array, z:Array, *args)->Array:
        
        mu, sigma = z[0], z[1]
        
        term1 = jnp.exp(-(x-mu)/sigma)/sigma
        term2 = jnp.exp(-jnp.exp(-(x-mu)/sigma))
        
        return term1*term2
    
    def pdf_frechet(self, x:Array, z:Array, *args)->Array:
        
        beta, lam = z[0], z[1]
        
        return (lam/beta)*((x/beta)**(-lam-1))*jnp.exp(-((x/beta)**(-lam)))
    
    def pdf_weibull(self, x:Array, z:Array, *args)->Array:
        
        beta, lam = z[0], z[1]
        
        return (lam/beta)*((x/beta)**(lam-1))*jnp.exp(-((x/beta)**(lam)))
    
    def pdf_pareto(self, x:Array, z:Array, *args)->Array:
        
        theta, alpha = z[0], z[1]
        
        return theta*(alpha**theta)*(x**(-(theta+1)))
    
    def pdf_power_function(self, x:Array, z:Array, *args)->Array:
        
        theta, alpha = z[0], z[1]
        
        return theta*(alpha**(-theta))*(x**(theta-1))
    
    def pdf_power_function(self, x:Array, z:Array, *args)->Array:
        
        theta, alpha = z[0], z[1]
        
        return theta*(alpha**(-theta))*(x**(theta-1))















    
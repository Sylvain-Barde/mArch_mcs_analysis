# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 11:38:17 2023

@author: Sylvain Barde, University of Kent

Implements multivariate ARCH estimation of the following specifications:
    -The Constant Conditional Correlation model (Bollerslev 1990)
    -The Dynamic Conditional Correlation (1,1) model (Engle 2002)
    -The Assymmetric Dynamic Conditional Correlation (1,1) model (Cappiello et.
     al. 2006)

This is implemented as a wrapper around the existing univariate 'arch' python 
toolbox: 
    https://pypi.org/project/arch/
    https://github.com/bashtage/arch
    https://bashtage.github.io/arch/    

Requires the following packages:

    arch
    numpy  (already a dependency of arch)
    pandas (already a dependency of arch)
    scipy  (already a dependency of arch)

Classes:

    mArch

Utilities:

    format_float_fixed
    
The toolbox contains code derived from the arch toolbox, Copyright (c) 2017 
Kevin Sheppard. All rights reserved.

Redistributions of source code must retain the above copyright notice, this 
list of conditions and the following disclaimers.

Redistributions in binary form must reproduce the above copyright notice, this 
list of conditions and the following disclaimers in the documentation and/or 
other materials provided with the distribution.

Neither the names of Kevin Sheppard, nor the names of its contributors may be
used to endorse or promote products derived from this Software without specific
prior written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE
SOFTWARE.

"""

import numpy as np
import pandas as pd
from arch.univariate import ConstantMean, StudentsT
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import norm
from scipy.linalg import sqrtm
from statsmodels.tools.numdiff import approx_fprime, approx_hess
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.summary import fmt_params

def format_float_fixed(x: float, max_digits: int = 10, decimal: int = 4) -> str:
    """Formats a floating point number so that if it can be well expressed
    in using a string with digits len, then it is converted simply, otherwise
    it is expressed in scientific notation
    
    Copied from arch toolbox source to ensure comparability with univariate 
    arch outputs.
    """
    # basic_format = '{:0.' + str(digits) + 'g}'
    if x == 0:
        return ("{:0." + str(decimal) + "f}").format(0.0)
    scale = np.log10(np.abs(x))
    scale = np.sign(scale) * np.ceil(np.abs(scale))
    if scale > (max_digits - 2 - decimal) or scale < -(decimal - 2):
        formatted = ("{0:" + str(max_digits) + "." + str(decimal) + "e}").format(x)
    else:
        formatted = ("{0:" + str(max_digits) + "." + str(decimal) + "f}").format(x)
    return formatted


class mArch:
    """
    Main module class for the package. See function-level help for details.

    Attributes:
        rt (ndarray):
            returns data for estimation
        Tfull (int):
            Number of time-series observations in full data
        N (int):
            Number of multivariate series
        varNames (list):
            List of names for the multivariate series
        init (dict):
            Default initial values for multivariate parameters
        bounds (dict):
            Parameter bounds for multivariate parameters
        constraints (dict):
            Constraints for bounds for multivariate parameters
        paramNames (dict):
            Parameter names for multivariate parameters
        T (int):
            Number of observations in re-sized data
        u (ndarray):
            array of univariate arch toolbox residuals
        sigma (ndarray):
            array of univariate arch toolbox estimated volatilities
        sigmaFs (ndarray):
            array of univariate arch toolbox forecasted volatilities
        archModel (list)
            list of univariate arch toolbox volatility models
        archLogLike (float)
            sum of univariate arch toolbox log-likelihoods
        archParams (list)
            list of univariate arch toolbox parameter estimates
        archResults (list)
            list of univariate arch toolbox fitted arch objects
        archForecast (list)
            list of univariate arch toolbox forecast arch objects
        uTilde (ndarray)
            array of normalised univariate arch toolbox residuals
        Qbar (ndarray)
            multivariate covariance matrix
        Rbar (ndarray)
            multivariate correlation matrix
        eta (ndarray)
            array of masked normalised univariate arch toolbox residuals
            (= uTilde where uTilde >0, 0 otherwise)
        Abar (ndarray)
            multivariate correlation matrix for the masked normalised residuals
        delta (float)
            largest eigenvalue of Abar, used for DCCA constraint
        multivarType (str)
            multivariate model to be estimated
        multivarResults (dict)
            dictionary of multivariate arch parameter estimation outputs
        errors (str)
            distribution to be used for volatility innovations
        numVolatilityParams (int)
            number of parameters in volatility model

    Methods:
        __init__ :
            Initialises an empty instance of the multivariate arch class
        _trimNan:
            Remove NAN values
        _archFit:
            Fit univariate ARCH processes
        _archForecast:
            Forcast univariate ARCH values
        _computeR:
            Compute time-varying multivariate correlation matrix
        _covLogLike:
            Return log-likelihood of the multivariate covariance matrices
        setArch:
            Setup the multivariate arch model
        fit
            Fit the multivariate arch model
        forecast
            Forecast values uing a fitted multivariate arch model
        summary
            Display estimation summary for a fitted multivariate arch model
            
    """    
    
    def __init__(self, data):
        """
        Initialises an empty instance of the multivariate arch class

        Parameters
        ----------
        data : ndarray
            2D ndarray containing the data. Structure is:
                
                num obs x num series
                
        Returns
        -------
        None.

        """
        
        # Set dataset to be estimated
        self.rt = data
        self.Tfull, self.N = data.shape
        if isinstance(data, pd.DataFrame):
            self.varNames = [col for col in data.columns]
        else:
            self.varNames = []
            for i in range(self.N):
                self.varName.append('var_{:d}'.format(i))
        
        # Preload estimation settings for possible options
        self.init = {
            'naive':np.empty(shape=(0)),
            'ccc':np.empty(shape=(0)),
            'dcc':np.asarray([0.45, 0.45]),
            'dcca':np.asarray([0.4, 0.4, 0]),
            'Normal':np.empty(shape=(0)),
            'Student':np.asarray([5])
            }
        
        self.bounds = {
            'naive':np.empty(shape=(0, 2)),
            'ccc':np.empty(shape=(0, 2)),
            'dcc':np.concatenate((np.asarray([0,1])[None,:],
                                  np.asarray([0,1])[None,:]),
                                    axis = 0),
            'dcca':np.concatenate((np.asarray([0,1])[None,:],
                                   np.asarray([0,1])[None,:],
                                   np.asarray([0,1])[None,:]),
                                    axis = 0),
            'Normal':np.empty(shape=(0, 2)),
            'Student':np.asarray([2.05,500])[None,:]
            }
        
        self.constraints = {
            'dcc':{'type':'ineq', 
                   'fun': lambda params: 1 - sum(params[0:2])},
            'dcca':{'type':'ineq', 
                    'fun': lambda params: (1 - sum(params[0:2])
                                             - self.delta*params[2])}
            }
        self.paramNames = {
            'dcc': ["alpha_m", "beta_m"],
            'dcca': ["alpha_m", "beta_m", "gamma_m"],
            'Normal': [],
            'Student':["nu"]
            }

        
    def _trimNan(self, X):
        """
        Remove NAN values
        Internal utility, not meant to be called outside of class

        Parameters
        ----------
        X : ndarray
            an array of data potentially containing NANs.

        Returns
        -------
        ndarray
            The input array with NANs trimmed out.
            
        """
    
        axisTuple = tuple(range(1,X.ndim))
        
        return X[~np.isnan(X).any(axis = axisTuple)]
        
    def _archFit(self, **kwargs):
        """
        Fit univariate ARCH processes
        
        Completes the first step of the multivariate arch estimation, which 
        is estimation of the univariate arch models. This method iterates 
        univariate estimation of an arch toolbox model for each individual
        series in the multivariate data, saves the set of univariate estimates
        and computes the average multivariate covariance and correlation.
        
        Internal utility, not meant to be called outside of class

        Parameters
        ----------
        **kwargs : dictionary
            Dictionary of named keywords passed to the arch toolbox .

        Returns
        -------
        None.

        """
        
        # Step 1 of the estimation, simply univariate ARCH/GARCH process
        
        # Adjust sample length if needed (if last obs is used)
        if 'last_obs' in kwargs:

            self.T = kwargs['last_obs']
        else:
            self.T = self.Tfull
        
        # For the volatility model, based on univariate estimation
        paramList = []
        resList = []
        self.u = np.zeros([self.Tfull, self.N])
        self.sigma = np.zeros([self.Tfull, self.N])
        self.archLogLike = 0

        for i in range(self.N):
            print('Univariate estimation of {:s}'.format(
                self.archModel[i].y.name))
            resList.append(self.archModel[i].fit(**kwargs))
            paramList.append(resList[-1].params)
            self.u[:,i] = resList[-1]._resid
            self.sigma[:,i] = resList[-1]._volatility**0.5
            self.archLogLike += resList[-1].loglikelihood
            
        # Set properties to class 
        self.archParams = paramList
        self.archResults = resList
        self.uTilde = self.u/self.sigma
        self.Qbar = np.cov(self._trimNan(self.uTilde), rowvar=False)
        self.Rbar = np.corrcoef(self._trimNan(self.uTilde), rowvar=False)
        self.eta = self.uTilde*np.where(
            np.where(~np.isnan(self.uTilde),self.uTilde,0) < 0,1,0)
        self.Abar = np.cov(self._trimNan(self.eta), rowvar=False) 

        QbarInvRoot = np.linalg.inv(sqrtm(self.Qbar))
        self.delta = max(np.linalg.eig(
                        QbarInvRoot @ self.Abar @ QbarInvRoot
                        )[0])

    def _archForecast(self, **kwargs):
        """
        Forcast univariate ARCH values
        
        Completes the first step of the multivariate arch forecating process by 
        forecasting univariate arch models. This method iterates 
        univariate forecasting of the arch toolbox model for each individual
        series in the multivariate data, saves the set of univariate forecasts.
        
        Internal utility, not meant to be called outside of class

        Parameters
        ----------
        **kwargs : dictionary
            Dictionary of named keywords passed to the arch toolbox .

        Returns
        -------
        None.

        """
        
        # Step 1 of the forecast, simply univariate ARCH/GARCH process
        predList = []
        sigmaList = []
        
        for i in range(self.N):
            predList.append(self.archResults[i].forecast(**kwargs))
            sigmaList.append(predList[-1].variance**0.5)
            
        self.archForecast = predList
        self.sigmaFs = sigmaList
        
        
    def _computeR(self,params,uTilde,eta):
        """
        Compute time-varying multivariate correlation matrix
        This function calculates the correlation matrices for DCC and DCCA
        specifications for a given set of univariate residuals and average
        correlation matrix

        Internal utility, not meant to be called outside of class

        Parameters
        ----------
        params : ndarray
            array of multivariate volatility model parameters.
        uTilde : ndarray 
            array of normalised univariate arch toolbox residuals.
        eta : ndarray
            array of masked normalised univariate arch toolbox residuals
            (= uTilde where uTilde >0, 0 otherwise)

        Returns
        -------
        R : ndarray
            3D ndarray containg the time-variying correlation matrices.
            Structure is:
               
               num obs x num series x num series

        """
        
        psiTilde = uTilde[:,:,None] @ uTilde[:,None,:]
        ATilde = eta[:,:,None] @ eta[:,None,:]
        
        a = params[0]
        b = params[1]
        if self.multivarType == 'dcca':
            g = params[2]
        else:
            g = 0
        
        Q = np.zeros(psiTilde.shape)
        Q[0,:,:] = self.Qbar
        
        for t in range(1,uTilde.shape[0]):
            Q[t,:,:] = (
                (1 - a - b)*self.Qbar 
                + a*psiTilde[t-1,:,:] 
                + b*Q[t-1,:,:]
                + g*(ATilde[t-1,:,:] - self.Abar)
                )
                    
        # Note: abs function protects against any negative variance in Q
        # Clearly, this suggests a problem has occured, and the negative 
        # Q will carry over to a negative R entry, but this avoids the 
        # negative power warning and creation of a NaN.
        with np.errstate(invalid='raise'):
        
            try: # Standard formula
                InvRootQ = np.einsum('ji,ki->jik', 
                                      np.einsum('ijj->ij',Q)**-0.5, 
                                      np.eye(Q.shape[1], dtype= uTilde.dtype))
            except: # if negative value, take absolute value of diagonal 
                InvRootQ = np.einsum('ji,ki->jik', 
                                      np.abs(np.einsum('ijj->ij',Q))**-0.5, 
                                      np.eye(Q.shape[1], dtype= uTilde.dtype))
            
        R = np.matmul(InvRootQ,
                      np.matmul(Q,
                                InvRootQ))
        
        return R
                        
    def _covLogLike(self, params, individual = False):
        """
        Return log-likelihood of the multivariate covariance matrices
        Used as the objective function of the second stage of the mArch 
        estimation, which is the estimation of the dynamic multivariate 
        conditional correlation process.
        
        Internal utility, not meant to be called outside of class

        Parameters
        ----------
        params : ndarray
            array of multivariate volatility model parameters.
        individual : boolean, optional
            Flag for return type. 
            - False makes the method returns the aggregate log-likelihood 
            - True makes the method returns the observation-level contributions
              to the log-likelihood 
            The default is False.

        Returns
        -------
        float or ndarray, depending on value of 'individual'
            Aggregate log-likelihood or vector of log-likelihood contributions
            (Depending on 'individual' flag)

        """
        
        # Only compute on non-NAN portions (allows for slicing)
        uTilde = self._trimNan(self.uTilde)
        eta = self._trimNan(self.eta)
        
        R = self._computeR(params,
                           uTilde,
                           eta)
        
        # Attach likelihood to class, call based on error model
        if self.errors == 'Normal':
            logLike = -0.5*(
                -( uTilde[:,None,:]@ uTilde[:,:,None]).squeeze()
                + np.linalg.slogdet(R)[1]
                + np.matmul(uTilde[:,None,:],
                            np.matmul(np.linalg.inv(R),
                                      uTilde[:,:,None])
                            ).squeeze()
                )
            
        elif self.errors == 'Student':
            nu = params[-1]
            logLike = (gammaln((nu + self.N) / 2) - gammaln(nu / 2)
                       - 0.5*np.log(np.pi*(nu-2))*self.N
                       - 0.5*np.linalg.slogdet(R)[1]
                       - 0.5*(self.N + nu)*np.log(1 + 
                                  np.matmul(uTilde[:,None,:],
                                            np.matmul(np.linalg.inv(R),
                                                      uTilde[:,:,None])
                                            ).squeeze()/(nu-2)
                            )
                )
        
        if individual:
            return logLike
        else:
            return sum(logLike) + self.archLogLike
        
        
    def setArch(self, volatilityModel, errors = 'Normal', multivar = 'dcc'):
        """
        Setup the multivariate arch model
        Configures the multivariate volatility model prior to estimation 

        Parameters
        ----------
        volatilityModel : instance of an arch toolbox volatility model
            The volatility model to be used for the first stage univariate arch
            estimation of the CCC/DCC/DCCA soecification.
            Examples: ARCH, GARCH, FIGARCH, etc. See arch documentation for
            further options.
        errors : str, optional
            Flag for the error type to be used in the estimation. Options are
            'Normal' and 'Student'. The default is 'Normal'.
        multivar : str, optional
            Flag for the specification of the multivariate correlation matrix.
            Options are:
                -'Naive': No multivariate correlation (Identity matrix)
                -'ccc': Constant Conditional Correlation model
                -'dcc': Dynamic Conditional Correlation model
                -'dcca': Assymmetric Dynamic Conditional Correlation model
            The default is 'dcc'.

        Returns
        -------
        None.

        """
        
        # Set error option, checking specification
        errorList = ['Normal', 'Student']
        if errors not in errorList:
            print(""" 'errors' parameter incorrectly specified. Set to:
                      - 'Normal' for multivariate Normal
                      - 'Student' for multivariate Student""")
            return
        else:
            self.errors = errors
            
        # Set multivariate option, checking specification
        multivarList = ['naive','ccc','dcc','dcca']
        if multivar not in multivarList:
            print(""" 'multivar' parameter incorrectly specified. Set to:
                      - 'naive' for a naive (uncorrelated) model
                      - 'ccc' for constant conditional correlation
                      - 'dcc' for dynamic conditional correlation
                      - 'dcca' for asymmetric dynamic conditional correlation""")
            return
        else:
            self.multivarType = multivar
        
        # Set volatility process
        self.numVolatilityParams = volatilityModel.num_params + 1
        modelList = []
        for i in range(self.N):
            mod = ConstantMean(self.rt.iloc[:,i])
            mod.volatility = volatilityModel
            # Set unit-level student dist. when multivariate correlation is not
            # estimated
            if ((self.multivarType == 'naive' or self.multivarType == 'ccc') 
                and self.errors == 'Student'):
                mod.distribution = StudentsT()
            
            modelList.append(mod)
        
        self.archModel = modelList
        
            
    def fit(self, **kwargs):
        """
        Fit the multivariate arch model
        Performs two-stage estimation of the multivariate volatility model
        - Stage 1: Series-wise estimation of a univariate volatility model 
          specified during .setArch(). This is carried out using the arch 
          toolbox, via the ._archFit() method.
        - Stage 2: Estimation of the multivariate volatility model using
          sequential least squares quadratic programming (SLSQP)
        
        Note: Estimation is onle carried out for the 'dcc' and 'dcca' 
        pecifications. 'Naive' and 'ccc' used fixed correlation matrices, so
        no estimation is required.

        Parameters
        ----------
        **kwargs : dictionary
            Dictionary of named keywords passed to the arch toolbox. These are: 
            - The estimation keywords that need to be passed to the univariate
              volatility model specified in the .setArch() method. See arch 
              documentation for further options.
            - An additional 'init' keyword containing user-specified initial
              values for the multivariate parameters. Optional, the toolbox
              provide default initial values. However, changing initial values
              can help when the second stage extimation fails to converge.

        Returns
        -------
        paramEst : scipy.minimize output structure
            The optimisation result obtained by maximising the log-likelihood
            of the multivariate model. This is carried out using SLSQP in 
            scipy.minimize()

        """
        
        # Check that a specification exists
        if not hasattr(self, 'multivarType') and not hasattr(self, 'errors'):
            print('Error - No multivariate model specified')
            print("Use the '.setArch() method to set the multivariate model")
            return
        
        # Pre-process - extract any user-provided initial values from kwargs
        # or generate defaults
        if 'init' in kwargs:
            init = np.asarray(kwargs['init'])
            del kwargs['init']
        else:
            init = np.concatenate((self.init[self.multivarType],
                                   self.init[self.errors]))
        
        # Step 1 - Univariate estimation of ARCH/GARCH process
        self._archFit(**kwargs)
        
        # Step 2 - Maximise multivariate likelihood
        if self.multivarType == 'dcc' or self.multivarType == 'dcca':
            print('Covariance process estimation - {:s}'.format(self.multivarType))

            bounds = np.concatenate((self.bounds[self.multivarType],
                        self.bounds[self.errors]),
                       axis = 0)
            
            paramEst = minimize(lambda *args: - self._covLogLike(*args), 
                            init, 
                            method = 'SLSQP',
                            bounds = bounds,
                            constraints = self.constraints[self.multivarType])
            
            self.multivarResults = {'params':paramEst.x}
            
            # Step 3 - parameter inference and results structure
            self.multivarResults['cov_type'] = self.archResults[0].cov_type
            hess = approx_hess(paramEst.x, self._covLogLike)
            hess /= self.rt.shape[0]
            inv_hess = -np.linalg.inv(hess)
            
            if self.archResults[0].cov_type == 'robust':
                kwargs = {"individual" : True}
                scores = approx_fprime(
                    paramEst.x, self._covLogLike, kwargs=kwargs
                    )  
                
                score_cov = np.cov(-scores.T)
                param_covar = (inv_hess.dot(score_cov).dot(inv_hess) 
                               / self.rt.shape[0])
                
            else:
                param_covar = inv_hess / self.rt.shape[0]
            
            self.multivarResults['param_covar'] = param_covar
            self.multivarResults['std_err'] = np.diag(param_covar)**0.5
            self.multivarResults['tvalues'] = (self.multivarResults['params']
                                               /self.multivarResults['std_err'])
         
        else:
            paramEst = []
            
        return paramEst
    
    
    def checkBoundary(self):
        """
        Run a post-estimation check of the DCC/DCCA parameter boundary 
        conditions. 
        This condition ensures that the resulting variance-covariance matrix Q 
        is positive definite. Failure to meet this condition will result in
        negative variances and negative losses.
        
        The conditions are:
            DCC:    1 - alpha_m - beta _m > 0
            DCCA:   1 - alpha_m - beta_m - gamma_m*self.delta > 0

        Note: these inequality constraints are imposed during the multivariate
        estimation in self.fit(), however SLSQP only imposes non-negativity, so
        a boundary solution (=0) can be accepted. In addition the numerical 
        tolerance on the solution means it is feasible for these boundary 
        solutions to fail (i.e. a very small negative value can be accepted).

        Returns
        -------
        boolean
            - True if the boundary condition on DCC/DCCA parameters holds or if
            the multivariate mode is CCC/Naive.
            - False otherwise

        """
        
        # Check there are already model estimates
        if not hasattr(self, 'multivarType') and not hasattr(self, 'errors'):
            print('Error - No multivariate model specified')
            print("Use the '.setArch() method to set the multivariate model,"+ 
                  " then the .fit() method to obtain estimates")
            return
        elif not hasattr(self, 'archResults'):
            print('Error - Multivariate model has not been estimated')
            print("Use the .fit() method to obtain estimates")
            return
        
        # Check conditions for positive-definiteness of Q for DCC / DCCA
        if self.multivarType == 'dcc' or self.multivarType == 'dcca':
            params = self.multivarResults['params']
            condVal = self.constraints[self.multivarType]['fun'](params)
            if condVal > 0:
                condStr = ' '
                condCheck = True
            else:
                condStr = 'not '
                condCheck = False
        
            print('Parameter boundary condition for {:s} is {:s}met:'.format(
                self.multivarType, condStr))
            print(' Condition value: {:e}'.format(condVal))
            return condCheck
            
        # For Naive & CCC, no need to check
        else:
            print('Multivariate mode is {:s}, no boundary condition'.format(
                self.multivarType))
            return True
        
        
    def forecast(self, **kwargs):
        """
        Forecast values uing a fitted multivariate arch model
        Performs two-stage forecasting with the multivariate volatility model
        - Stage 1: Series-wise forecasting with the univariate volatility model 
          specified during .setArch(). This is carried out using the arch 
          toolbox, via the ._archForecast() method.
        - Stage 2: Forecasting with the multivariate volatility model

        Note: the implementation assumes the arch toolbox default that 
        forecasts are aligned on the observation in which the forecast is 
        generated (aligned = "origin"), NOT with the targeted observation 
        (aligned = "target")

        Parameters
        ----------
        **kwargs : dictionary
            Dictionary of named keywords passed to the arch toolbox. These are: 
            - The forecasting keywords that need to be passed to the univariate 
              volatility model specified in the .setArch() method. See arch 
              documentation for further options.

        Returns
        -------
        forecastDict : dict
            A dictionary containing the multivariate covariance matrix 
            forecasts, with one entry per forecast horizon.

        """
        
        # Check there are already model estimates
        if not hasattr(self, 'multivarType') and not hasattr(self, 'errors'):
            print('Error - No multivariate model specified')
            print("Use the '.setArch() method to set the multivariate model,"+ 
                  " then the .fit() method to obtain estimates")
            return
        elif not hasattr(self, 'archResults'):
            print('Error - Multivariate model has not been estimated')
            print("Use the .fit() method to obtain estimates")
            return
        
        # Check for desired reindexing behaviour
        if 'reindex' in kwargs:
            if kwargs['reindex'] == False:
                trim = True
            else:
                trim = False
        else:
            trim = True
        
        
        if self.multivarType == 'dcc' or self.multivarType == 'dcca':
        
            # Get one-step-ahead value of R (used for all forecast horizons)
            # based on model residuals and sigmas calculated at last data point
            u = np.zeros([self.Tfull, self.N])
            sigma = np.zeros([self.Tfull, self.N])
            for i in range(self.N):
                fixedParams = self.archResults[i].params
                fixedRes = self.archModel[i].fix(fixedParams)
                u[:,i] = fixedRes.resid
                sigma[:,i] = fixedRes._volatility**0.5
                
            uTildeF = u/sigma
            etaF = uTildeF*np.where(
                np.where(~np.isnan(uTildeF),uTildeF,0) < 0,1,0)
            
            # Compute one-step-ahead forecasted R (shifted by one)
            # Entry in Rf[t] is calculated using uTilde[t-1], eta[t-1]
            # To ensure the entry in Rf[t] contains the t+1 forecast, generated
            # with all data from uTilde[t], eta[t] one lead must be taken for 
            # these inputs.
            params = self.multivarResults['params']
            paramSum = sum(params[0:2])
            Rf = self._computeR(params,
                                np.roll(uTildeF,-1,axis = 0),
                                np.roll(etaF,-1,axis = 0)
                                )
            
        elif self.multivarType == 'ccc':
            
            paramSum = 0  # Force Rf component in H calculation for be fixed
                          # at unconditional correlation Rbar
            Rf = self.Rbar
            
        elif self.multivarType == 'naive':
            
            paramSum = 1  # Will cancel out Rbar component in H calculation 
            Rf = np.einsum('ji,ki->jik', 
                      np.ones([self.Tfull, self.N]),
                      np.eye(self.N, dtype=np.float64)) #Rf set to identity     
    
        # Get univariate forecasts for specified horizons
        # This sets the 'self.sigmaFs' used afterwards
        kwargs['reindex'] = True
        self._archForecast(**kwargs) 
        
        # Calculate R-based forecast given sigmas, for each horizon
        horizons = [col for col in self.sigmaFs[0].columns]        
        forecastDict = {}
        k = 0
        for horizon in horizons:
            
            sigma = np.zeros([self.Tfull, self.N])
            for i in range(self.N):
                sigma[:,i] = self.sigmaFs[i][horizon]
                
            D = np.einsum('ji,ki->jik', 
                      sigma,
                      np.eye(sigma.shape[1], dtype=sigma.dtype))
            H = np.matmul(D,
                      np.matmul(self.Rbar*(1-paramSum**k) + Rf*paramSum**k,
                                D))            
            if trim:
                forecastDict[horizon] = self._trimNan(H)
            else:
                forecastDict[horizon] = H
                
            k+=1
        
        return forecastDict
                
        
    def summary(self, init = True, alpha = 0.05):
        """
        Display estimation summary for a fitted multivariate arch model
        Summary of estimation results. Re-uses the layout of the arch toolbox 
        for comparability.

        Parameters
        ----------
        init : boolean, optional
            Flag for including the univariate estimates in the summary.
            - Set to 'True' to display both univariate and multivariate results
            - Set to 'False' to only display multivariate results
            The default is True.
        alpha : float, optional
            Significance required for confidence intervals. The default is 0.05

        Returns
        -------
        None.

        """
        
        # Check if model has been fitted
        if not hasattr(self, 'archResults'):
            print("Error - Model has not yet been fitted.")
            print("Use the .fit() method to obtain estimates")
            return
        
        if init is True:
            for i in range(self.N):
                print(self.archResults[i].summary())
        
        # Calculate confidence intervals and P-values at alpha
        # Note: Code reuses the 'Summary' code of arch toolbox
        
        if hasattr(self, 'multivarResults'):
        
            self.multivarResults['pvalues'] = np.asarray(
                norm.sf(np.abs(self.multivarResults['tvalues'])) * 2, 
                        dtype=float)        
            
            cv = norm.ppf(1 - alpha / 2)
            self.multivarResults['conf_int'] = np.vstack(
                (self.multivarResults['params'] - cv*self.multivarResults['std_err'],
                 self.multivarResults['params'] + cv*self.multivarResults['std_err'])
                ).T
            
            # Format outputs and generate table
            conf_int_str = []
            for c in self.multivarResults['conf_int']:
                conf_int_str.append(
                    "["
                    + format_float_fixed(c[0], 7, 3)
                    + ","
                    + format_float_fixed(c[1], 7, 3)
                    + "]"
                )
            
            title = "Multivariate process estimation - {:s}".format(
                self.multivarType)
            header = ["coef", "std err", "t", "P>|t|", "95.0% Conf. Int."]
            stubs = (self.paramNames[self.multivarType] 
                     + self.paramNames[self.errors])
            table_vals = (
                self.multivarResults['params'],
                self.multivarResults['std_err'],
                self.multivarResults['tvalues'],
                self.multivarResults['pvalues'],
                conf_int_str,
            )
            
            formats = [(10, 4), (9, 3), (9, 3), (9, 3), (0, 0)]
            pos = 0
            param_table_data = []
            for _ in range(len(table_vals[0])):
                row = []
                for i, val in enumerate(table_vals):
                    if isinstance(val[pos], np.float64):
                        converted = format_float_fixed(val[pos], *formats[i])
                    else:
                        converted = val[pos]
                    row.append(converted)
                pos += 1
                param_table_data.append(row)
            
            table = SimpleTable(
                    param_table_data,
                    stubs=stubs,
                    txt_fmt=fmt_params,
                    headers=header,
                    title=title,
                )
    
            print(table)
        else:
            print('Multivariate mode is {:s}, no estimation required'.format(
                self.multivarType))

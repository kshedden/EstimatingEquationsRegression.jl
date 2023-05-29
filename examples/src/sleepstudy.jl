# ## Sleep study (linear GEE)

# The sleepstudy data are from a study of subjects experiencing sleep
# deprivation.  Reaction times were measured at baseline (day 0) and
# after each of several consecutive days of sleep deprivation (3 hours
# of sleep each night).  This example fits a linear model to the reaction
# times, with the mean reaction time being modeled as a linear function
# of the number of days since the subject began experiencing sleep
# deprivation.  The data are clustered by subject, and since the data
# are collected by time, we use a first-order autoregressive working
# correlation model.

using EstimatingEquationsRegression, RDatasets, StatsModels

slp = dataset("lme4", "sleepstudy");

## The data must be sorted by the group id.
slp = sort(slp, :Subject);

m1 = fit(GeneralizedEstimatingEquationsModel,
         @formula(Reaction ~ Days), slp, slp[:, :Subject],
         IdentityLink(), ConstantVar(), AR1Cor())

# The scale parameter (unexplained standard deviation).
sqrt(dispersion(m1.model))

# The AR1 correlation parameter.
corparams(m1.model)

# The results indicate that reaction times become around 10.5 units
# slower for each additional day on the study, starting from a baseline
# mean value of around 253 units.  There are around 47.8 standard
# deviation units of unexplained variation, and the within-subject
# autocorrelation of the unexplained variation decays exponentially with
# a parameter of around 0.77.

# There are several approaches to estimating the covariance of the
# parameter estimates, the default is the robust (sandwich) approach.
# Other options are the "naive" approach, the "md" (Mancl-DeRouen)
# bias-reduced approach, and the "kc" (Kauermann-Carroll) bias-reduced
# approach.  Below we use the Mancl-DeRouen approach.  Note that this
# does not change the coefficient estimates, but the standard errors,
# test statistics (z), and p-values are affected.

m2 = fit(GeneralizedEstimatingEquationsModel,
         @formula(Reaction ~ Days), slp, slp.Subject,
         IdentityLink(), ConstantVar(), AR1Cor(), cov_type="md")

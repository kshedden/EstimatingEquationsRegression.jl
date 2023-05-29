# ## Length of hospital stay

# Below we look at data on length of hospital stay for patients
# undergoing a cardiovascular procedure.  This example illustrates how
# the variance function can be changed to a non-standard form.  Modeling
# the variance as μ^p for 1<=p<=2 gives a Tweedie model, and when p=1 or
# p=2 we have a Poisson or a Gamma model, respectively.  For 1<p<2, the
# inference is via quasi-likelihood as the score equations solved by GEE
# do not correspond to the score function of the log-likelihood of the
# data (even when there is no dependence within clusters).  In this
# example, as is often the case, the parameter estimates and standard
# errors are not strongly sensitive to the variance model.

using GEE, RDatasets, GLM

azpro = dataset("COUNT", "azpro")

azpro[!, :Los] = Float64.(azpro[:, :Los])

azpro = sort(azpro, :Hospital)

m1 = fit(GeneralizedEstimatingEquationsModel,
         @formula(Los ~ Procedure + Sex + Age75), azpro, azpro[:, :Hospital],
         LogLink(), IdentityVar(), IndependenceCor())

m2 = fit(GeneralizedEstimatingEquationsModel,
         @formula(Los ~ Procedure + Sex + Age75), azpro, azpro[:, :Hospital],
         LogLink(), PowerVar(1.5), IndependenceCor())

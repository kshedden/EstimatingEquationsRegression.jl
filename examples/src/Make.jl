using Literate

#Literate.markdown("sleepstudy.jl", "..", execute=true)
#Literate.markdown("contraception.jl", "..", execute=true)
#Literate.markdown("hospitalstay.jl", "..", execute=true)

#Literate.markdown("scoretest_simstudy.jl", "..", execute=true)
#Literate.markdown("expectiles_simstudy.jl", "..", execute=true)

Literate.markdown("README.jl", "../.."; execute=true, flavor=Literate.CommonMarkFlavor())

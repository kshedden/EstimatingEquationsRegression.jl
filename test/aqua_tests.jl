
@testitem "run aqua tests" begin

    using Aqua

    # Many ambiguities in dependencies
    Aqua.test_all(EstimatingEquationsRegression)
    Aqua.test_ambiguities(EstimatingEquationsRegression)
    Aqua.test_unbound_args(EstimatingEquationsRegression)
    Aqua.test_deps_compat(EstimatingEquationsRegression)
    Aqua.test_undefined_exports(EstimatingEquationsRegression)
    Aqua.test_project_extras(EstimatingEquationsRegression)
    Aqua.test_stale_deps(EstimatingEquationsRegression)
    Aqua.test_deps_compat(EstimatingEquationsRegression)
    Aqua.test_piracies(EstimatingEquationsRegression)
    Aqua.test_persistent_tasks(EstimatingEquationsRegression)
end

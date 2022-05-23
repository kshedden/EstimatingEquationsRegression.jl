# Return a 2 x m array, each column of which contains the indices
# spanning one group; also return the size of the largest group.
function groupix(g::AbstractVector)

    if !issorted(g)
        error("Group vector is not sorted")
    end

    ii = Int[]
    b, mx = 1, 0
    for i = 2:length(g)
        if g[i] != g[i-1]
            push!(ii, b, i - 1)
            mx = i - b > mx ? i - b : mx
            b = i
        end
    end
    push!(ii, b, length(g))
    mx = length(g) - b + 1 > mx ? length(g) - b + 1 : mx
    ii = reshape(ii, 2, div(length(ii), 2))

    return tuple(ii, mx)
end

"""
    expand_ordinal(df, response; response_recoded, var_id, level_var)

Construct a dataframe from source `df` that converts an ordinal
variable into a series of binary indicators.  These indicators can
then be modeled using binomial GEE with the `OrdinalIndependenceCor`
working correlation structure.

`response` must be a column name in `df` containing an ordinal
variable.  For each threshold `t` in the unique values of this
variable an indicator that the value of the variable is `>= t` is
created.  The smallest unique value in `response` is omitted.  The
recoded variable is called `response_recoded` and a default name is
created if no name is provided.  A variable called `var_id` is created
that gives a unique integer label for each set of indicators derived
from a common observed ordinal value.  The threshold value used to
create each binary indicator is placed into the variable named
`level_var`.
"""
function expand_ordinal(
    df,
    response::Symbol;
    response_recoded::Union{Symbol,Nothing} = nothing,
    var_id::Symbol = :var_id,
    level_var::Symbol = :level_var,
)

    if isnothing(response_recoded)
        s = string(response)
        response_recoded = Symbol(s * "_recoded")
    end

    # Create a schema for the new dataframe
    s = [Symbol(x) => Vector{eltype(df[:, x])}() for x in names(df)]
    e = eltype(df[:, response])
    push!(
        s,
        response_recoded => Vector{e}(),
        var_id => Vector{Int}(),
        level_var => Vector{e}(),
    )

    # Create binary indicators for these thresholds
    levels = unique(df[:, response])[2:end]
    sort!(levels)

    ii = 1
    for i = 1:size(df, 1)
        for t in levels
            for j in eachindex(s)
                if s[j].first == var_id
                    push!(s[j].second, ii)
                elseif s[j].first == level_var
                    push!(s[j].second, t)
                elseif s[j].first == response_recoded
                    push!(s[j].second, df[i, response] >= t ? 1 : 0)
                else
                    push!(s[j].second, df[i, s[j].first])
                end
            end
        end
        ii += 1
    end

    return DataFrame(s)
end

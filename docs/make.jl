
push!(LOAD_PATH, "../src")
using Documenter, GEE

makedocs(sitename="GEE.jl", modules=[GEE], pages=["Home"=>"index.md"])

deploydocs(; repo="github.com/kshedden/GEE.jl",)


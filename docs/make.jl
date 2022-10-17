using Uniformization
using Documenter

DocMeta.setdocmeta!(Uniformization, :DocTestSetup, :(using Uniformization); recursive=true)

makedocs(;
    modules=[Uniformization],
    authors="Garrett Smith <gasmith@uni-potsdam.de> and contributors",
    repo="https://github.com/smith-garrett/Uniformization.jl/blob/{commit}{path}#{line}",
    sitename="Uniformization.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://smith-garrett.github.io/Uniformization.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/smith-garrett/Uniformization.jl",
    devbranch="main",
)

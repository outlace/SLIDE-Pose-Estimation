module SRP
using Distributions;
export SRPHash

function GenSimHashMat(n,m)
    A = Array{Float64, 2}(undef, n, m)
    c = [1,-1]
    #d = √3
    #p = [0.166666667,0.666666667,0.166666667]
    p = [0.5,0.5]
    r = Categorical(p)
    for i in 1:n
        for j in 1:m
            A[i,j] = c[rand(r)];
        end
    end
    A
end

function SRPHash(in_dim::Int64,n::Int64)
    mat = GenSimHashMat(n,in_dim)#randn(n,in_dim)
    function hash(x)
        if length(x) == in_dim
            r = mat * x
            r = map(x -> x > 0, r)
        else
            error("Dimension mismatch between input with length $(length(x)) and matrix with dimensions $(size(mat))")
        end
        r
    end
end

end
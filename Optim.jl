module Optim

export SGD, SGDM, step!

mutable struct SGD
   step::Float64
end

SGD() = SGD(0.001)

mutable struct SGDM
    step::Float64
    γ::Float64
    v_old
end

SGDM() = SGDM(0.001, 0.9, [])

mutable struct ADAM
    step
    mt
    vt
    β₁
    β₂
    ϵ
end

#outer constructor
function ADAM(thetaV::Vector,step=0.01,β₁=0.9,β₂=0.999,ϵ=10^(-8))
    #thetaV is a vector of parameter views
    mt = [zeros(size(t.parent)) for t in thetaV]
    vt = copy(mt)
    ADAM(step,mt,vt,β₁,β₂,ϵ)
end

function step!(opt::SGD,theta,grad)
    #theta is a vector of arrays (the parameters of the model
    #grad is a vector of arrays (the gradients for each of the parameters)
    #the parameters will be updated in place
    for (p,g) in zip(theta,grad)
       p .-= opt.step * g
    end
end

function step!(opt::SGDM,theta,grad)
    if isempty(opt.v_old)
        for t in theta
            push!(opt.v_old,zeros(size(t.parent)))
        end
    end
    i = 1
    for (t,g) in zip(theta,grad)
        old = opt.v_old[i]
        idx = t.indices
        r = SubArray(old,idx)
        r = (opt.γ * r .+ opt.step * g)
        t .-= r
    end
end

function step!(opt::ADAM,theta,grad)
    i = 1
    for (t,g) in zip(theta,grad)
        idx = t.indices
        mtV = SubArray(opt.mt[i],idx)
        vtV = SubArray(opt.vt[i],idx)
        mtV = (opt.β₁ * mtV) .+ (1 - opt.β₁)*g
        vtV = (opt.β₂ * vtV) .+ (1 - opt.β₂)*g.^2
        #bias correction
        mtV = mtV ./ (1 - opt.β₁)
        vtV = vtV ./ (1 - opt.β₂)
        r = (opt.step * mtV) ./ (.√(vtV) .+ opt.ϵ)
        t .-= r
    end
end

end
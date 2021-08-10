module SVM2
using Random
using Plots
using Convex, SCS, ECOS
using Distributions: MvNormal

export gen_test_data, svmtrain, plot_svm, probabilityvector

function gen_test_data()
    # Generate data for SVM classifier with L1 regularization.
    Random.seed!(3);
    n = 4;
    m = 10000;
    TEST = m;
    beta_true = [0.5, 0.5, 0, 0]
    offset = 80;
    sigma = 0.001;
    X = rand(MvNormal([75, 80, 65, 70], 10.0), m);
    X = X';
    Y = sign.(X * beta_true .- offset .+ sigma * randn(m,1));
    X_test = randn(TEST, n);
    Y_test = sign.(X_test * beta_true .- offset .+ sigma * randn(m,1));
    X, Y, X_test, Y_test, beta_true, offset
end

function fix_scale_vector(d::Int)
    beta = Variable(d, Positive())
    add_constraint!(beta, sum(beta) == 1)
    return beta
end

function svmtrain2(X, Y, X_test, Y_test, beta_true, v_true, C)
    # Form SVM with L1 regularization problem.
    (m, n) = size(X)
    m_test =  size(X_test)[1]
    @show m,m_test

    # Compute a trade-off curve and record train and test error.
    len_C= length(C)
    train_error = zeros(len_C);
    test_error = zeros(len_C);
    beta_error = zeros(len_C);
    v_error = zeros(len_C);
    beta_vals = zeros(n, len_C);
    λ_record = zeros(len_C);

    beta = fix_scale_vector(n) # add_constraint!(beta, sum(beta) == 1)
    v = Variable();
    reg = norm(beta, 1);
    # pos(x::AbstractExpr) = max(x, Constant(0, Positive()))
    # hinge_loss(x::AbstractExpr) = pos(1 - x)
    # loss = sum(pos(1 - Y .* (X*beta + v))); # sign or ||,  transit 1, non -1
    loss = sum(pos(1 - Y .* (X*beta + v)));

    for i in 1:length(C)
        λ = C[i]/sqrt(m)
        problem = minimize(loss/m + λ*reg);
        # solve!(problem, () -> ECOS.Optimizer(verbose=0));
        solve!(problem, () -> SCS.Optimizer(verbose=0));

        # Evaluation
        λ_record[i] = λ
        train_error[i] = sum(float(Y .!= sign.(evaluate(X*beta + v))))/m * 100;
        test_error[i] = sum(float(Y_test .!= sign.(evaluate(X_test*beta + v))))/m_test * 100;
        beta_vals[:, i] =  evaluate(beta);
        beta_error[i] = norm(beta.value - beta_true)
        v_error[i] = abs(v_true-v.value)
    end

    [v_error, beta_error, train_error, test_error]
end


end

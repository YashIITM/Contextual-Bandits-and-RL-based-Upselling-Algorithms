import numpy as np

def make_design_matrix(n_trial, n_arms, n_feature):
    """
    Returns the design matrix ofsize n_trial*n_arms*n_feature
    """
    available_arms = np.arange(n_arms)
    X = np.array([[np.random.uniform(low=0, high=1, size=n_feature) for _ in available_arms] for _ in np.arange(n_trial)])
    return X

def make_theta(n_arms, n_feature, best_arms, bias = 1):
    """
    Returns the parameter matrix ofsize n_arms*n_feature
    """
    true_theta = np.array([np.random.normal(size=n_feature, scale=1/4) for _ in np.arange(n_arms)])
    true_theta[best_arms] = true_theta[best_arms] + bias
    return true_theta

def generate_reward(arm, x, theta, scale_noise = 1/10):
    signal = theta[arm].dot(x)
    noise = np.random.normal(scale=scale_noise)
    return (signal + noise)

def make_regret(payoff, oracle):
    return np.cumsum(oracle - payoff)


def GetRealReward(context: np.ndarray, theta: np.ndarray,scale_noise = 1/10) -> np.ndarray:
    """Given the context, return the realized reward

    Args:
        context (np.ndarray): An np.ndarray whose shape is (K, d), each column represents a context of an arm
        theta is true theta(np.ndarray): The parameter of this reward function

    Returns:
        reward: an np.ndarray whose shape is (K,), reward = context^T A^T A context + N(0, 0.05^2)
    """
    rewards = []
    for arm in range(theta.shape[0]):
        signal = theta[arm].dot(context[arm])
        noise = np.random.normal(scale=scale_noise)
        rewards.append(noise+signal)
    return np.array(rewards)


def linUCB_disjoint(alpha, X, generate_reward, true_theta):
    print("linUCB disjoint with exploration parameter alpha: ", alpha)
    n_trial, n_arms, n_feature = X.shape
    # 1. Initialize object
    # 1.1. Output object
    arm_choice = np.empty(n_trial) # store arm choice (integer) for each trial
    r_payoff = np.empty(n_trial) # store payoff (float) for each trial
    theta = np.empty(shape=(n_trial, n_arms, n_feature)) # record theta over each trial (n_arms, n_feature) per trial
    p = np.empty(shape = (n_trial, n_arms)) # predictions for reward of each arm for each trial
    # 1.2 Intermediate object
    A = np.array([np.diag(np.ones(shape=n_feature)) for _ in np.arange(n_arms)])
    b = np.array([np.zeros(shape=n_feature) for _ in np.arange(n_arms)])
    # 2. Algo
    for t in np.arange(n_trial):
        # Compute estimates (theta) and prediction (p) for all arms
        for a in np.arange(n_arms):
            inv_A = np.linalg.inv(A[a])
            theta[t, a] = inv_A.dot(b[a])
            p[t, a] = theta[t, a].dot(X[t, a]) + alpha * np.sqrt(X[t, a].dot(inv_A).dot(X[t, a]))
        # Choosing best arms
        chosen_arm = np.argmax(p[t])
        x_chosen_arm = X[t, chosen_arm]
        r_payoff[t] = generate_reward(arm=chosen_arm, x=x_chosen_arm, theta=true_theta)

        arm_choice[t] = chosen_arm
        
        # update intermediate objects (A and b)
        A[chosen_arm] += np.outer(x_chosen_arm, x_chosen_arm.T)
        b[chosen_arm] += r_payoff[t]*x_chosen_arm
    return dict(theta=theta, p=p, arm_choice=arm_choice, r_payoff=r_payoff)


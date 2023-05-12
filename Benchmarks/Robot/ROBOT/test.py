# import numpy as np

# trial1 = [[1,4,3], [5,5,6], [9,8,9]]
# trial2 = [[1,2,3], [4,5,6], [7,8,9]]

# # multiply trials
# trial1 = np.array(trial1)
# # print(trial1)
# # trial2 = np.array(trial2)
# # print(trial2)
# # trial1 = np.dot(trial1, trial2)
# # print(trial1)

# #inverse of trial1
# trial1 = np.linalg.inv(trial1)
# print(trial1)

from control import dare, dlqr
import numpy as np

if __name__ == "__main__":

    # std::array<std::array<T,3>,3> A{{{1,0.1,0},{0,1,0.1},{0,0,0}}};

    # std::array<std::array<T,3>,3> A_approx{{{1,0.1,0},{0,1,0.1},{0,0,0.001}}};

    # std::array<std::array<T,3>,3> B{{{1,0,0},{0,1,0},{0,0,1}}};

    # std::array<std::array<T,3>,3> Q{{{1,0,0},{0,0.1,0},{0,0,0.1}}};

    # std::array<std::array<T,3>,3> R{{{1000,0,0},{0,1000,0},{0,0,1}}};

    A = np.array([[1,0.1,0],[0,1,0.1],[0,0,0]])
    A_approx = np.array([[1,0.1,0],[0,1,0.1],[0,0,0.001]])
    B = np.array([[1,0,0],[0,1,0],[0,0,1]])
    Q = np.array([[1,0,0],[0,0.1,0],[0,0,0.1]])
    R = np.array([[100,0,0],[0,100,0],[0,0,1]])
    N = np.array([[0,0,0],[0,0,0],[0,0,0]])

    # X,L,G = dare(A, B, Q, R)

    # print(X)
    # print(L)
    # print(G)

    K,S,E = dlqr(A, B, Q, R, N)

    print(K)



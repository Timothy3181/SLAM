#include <ctime>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>

#define MATRIX_SIZE = 50

int main()
{
    // 3x3 Matrix
    Eigen::Matrix<float, 3, 3> matrix1;
    
    // 3 dimension vector
    // equal to Eigen::Matrix<double, 3, 1>
    Eigen::Vector3d vector1;

    // 3x3 Matrix with the type of double
    Eigen::Matrix3d matrix2;
    // and which can be initialized as a zero matrix
    matrix2 = Eigen::Matrix3d::Zero();

    // unsure the size of the matrix
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix3;
    // which is the same as
    Eigen::MatrixXd matrix4;

    // the matrix can be initialized by the method of cpp stream
    matrix1 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    std::cout << "The 3x3 matrix is: \n" << matrix1 << std::endl;
    // also it can be visited by loop
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << matrix1(i, j) << " ";
        }
        std::cout << std::endl;
    }

    // multiply vector and matrix
    Eigen::Vector3d v_3d;
    v_3d << 3, 2, 1;
    Eigen::Matrix<float, 3, 1> vd_3d;
    vd_3d << 4, 5, 6;
    // cannot multiply directly, because of the different types
    auto result1 = matrix1.cast<double>() * v_3d;
    std::cout << "[1, 2, 3\n4, 5, 6\n7, 8, 9] * [3, 2, 1] = " << result1.transpose() << std::endl;
    auto result2 = matrix1 * vd_3d;
    std::cout << "[1, 2, 3\n4, 5, 6] * [4, 5, 6] = " << result2.transpose() << std::endl;
    
    // a random matrix, and the method to get transpose, sum, trace, times, inverse and det
    Eigen::Matrix3d matrix_random = Eigen::Matrix3d::Random();
    std::cout << "random result: " << matrix_random << std::endl;
    std::cout << "transpose: " << matrix_random.transpose() << std::endl;
    std::cout << "sum: " << matrix_random.sum() << std::endl;
    std::cout << "trace: " << matrix_random.trace() << std::endl;
    std::cout << "times: (10) " << 10 * matrix_random << std::endl;
    std::cout << "inverse: " << matrix_random.inverse() << std::endl;
    std::cout << "det: " << matrix_random.determinant() << std::endl;
    
    // the special value
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(matrix_random.transpose() * matrix_random);
    std::cout << "Eigen values = " << eigen_solver.eigenvalues() << std::endl;
    std::cout << "Eigen vectors = " << eigen_solver.eigenvectors() << std::endl;

    // solve an equation of linear algebra
    Eigen::Matrix<double, 50, 50> matrix50 = Eigen::MatrixXd::Random(50, 50);
    matrix50 = matrix50 * matrix50.transpose();
    Eigen::Matrix<double, 50, 1> v_Nd = Eigen::MatrixXd::Random(50, 1);
    // timer
    std::clock_t time_stt = clock();
    // get the inverse directly
    Eigen::Matrix<double, 50, 1> x = matrix50.inverse() * v_Nd;
    std::cout << "time of the normal inverse is "
        << 1000 * (clock() - time_stt) / static_cast<double>(CLOCKS_PER_SEC) << "ms" << std::endl;
    std::cout << "x = " << x.transpose() << std::endl;
    // solve by separating the matrix usually, for example QR Separate Method
    time_stt = clock();
    x = matrix50.colPivHouseholderQr().solve(v_Nd);
    std::cout << "time of the QR decomposition is "
        << 1000 * (clock() - time_stt) / static_cast<double>(CLOCKS_PER_SEC) << "ms" << std::endl;
    std::cout << "x = " << x.transpose() << std::endl;
    // cholesky
    time_stt = clock();
    x = matrix50.ldlt().solve(v_Nd);
    std::cout << "time of the ldlt decomposition is "
        << 1000 * (clock() - time_stt) / static_cast<double>(CLOCKS_PER_SEC) << "ms" << std::endl;
    std::cout << "x = " << x.transpose() << std::endl;

    return 0;
}

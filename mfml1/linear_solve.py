import math
def print_matrix(matrix, title="Matrix"):
    """Display a matrix in a readable format."""
    print(f"\n{title}:")
    
    for row in matrix:
        print("  ".join(f"{value:10.4f}" for value in row))


def validate_input(A, b):
    """Validate coefficient matrix and RHS vector."""

    # Check that A is not empty
    if not A or not all(A):
        raise ValueError("Coefficient matrix cannot be empty.")

    n = len(A)

    # Check whether A is square
    if any(len(row) != n for row in A):
        raise ValueError(
            "Invalid matrix dimensions: coefficient matrix must be square."
        )

    # Check RHS dimensions
    if len(b) != n:
        raise ValueError(
            "Invalid matrix dimensions: RHS vector size "
            "must match the number of rows in A."
        )


def create_augmented_matrix(A, b):
    """Create and return the augmented matrix [A | b]."""

    return [
        [float(value) for value in row] + [float(b[i])]
        for i, row in enumerate(A)
    ]


def gaussian_elimination(A, b, tolerance=1e-10):
    """
    Perform Gaussian elimination with partial pivoting.

    Returns:
        Upper triangular augmented matrix.
    """

    validate_input(A, b)

    augmented = create_augmented_matrix(A, b)
    n = len(augmented)

    for pivot_col in range(n):

        # Partial pivoting:
        # Find row with largest absolute pivot value
        pivot_row = max(
            range(pivot_col, n),
            key=lambda row: abs(augmented[row][pivot_col])
        )

        # Check for zero / extremely small pivot
        if abs(augmented[pivot_row][pivot_col]) < tolerance:
            continue

        # Swap rows
        if pivot_row != pivot_col:
            augmented[pivot_col], augmented[pivot_row] = (
                augmented[pivot_row],
                augmented[pivot_col]
            )

        # Eliminate values below pivot
        for row in range(pivot_col + 1, n):

            if abs(augmented[row][pivot_col]) < tolerance:
                augmented[row][pivot_col] = 0.0
                continue

            factor = (
                augmented[row][pivot_col]
                / augmented[pivot_col][pivot_col]
            )

            for col in range(pivot_col, n + 1):
                augmented[row][col] -= (
                    factor * augmented[pivot_col][col]
                )

            # Avoid tiny floating-point values
            if abs(augmented[row][pivot_col]) < tolerance:
                augmented[row][pivot_col] = 0.0

    # Check for singular or inconsistent system
    for row in augmented:
        coefficients = row[:-1]
        rhs = row[-1]

        all_zero = all(abs(value) < tolerance for value in coefficients)

        if all_zero and abs(rhs) >= tolerance:
            raise ValueError(
                "System is inconsistent and has no solution."
            )

        if all_zero and abs(rhs) < tolerance:
            raise ValueError(
                "System is singular and does not have a unique solution."
            )

    return augmented


def back_substitution(upper_matrix, tolerance=1e-10):
    """Solve an upper triangular system using back substitution."""

    n = len(upper_matrix)
    x = [0.0] * n

    for i in range(n - 1, -1, -1):

        pivot = upper_matrix[i][i]

        if abs(pivot) < tolerance:
            raise ValueError(
                "Zero pivot encountered during back substitution."
            )

        sum_known = 0.0

        for j in range(i + 1, n):
            sum_known += upper_matrix[i][j] * x[j]

        x[i] = (upper_matrix[i][n] - sum_known) / pivot

    return x


def verify_solution(A, b, x, tolerance=1e-8):
    """Verify whether Ax is approximately equal to b."""

    calculated = []

    for row in A:
        value = math.ceil(round(sum(row[j] * x[j] for j in range(len(x))),2))
        calculated.append(value)

    verification = all(
        abs(calculated[i] - b[i]) < tolerance
        for i in range(len(b))
    )

    return verification, calculated


def solve_system(A, b):
    """Solve a complete system using Gaussian elimination."""

    print_matrix(A, "Input Coefficient Matrix")

    print("\nRHS Vector:")
    print(b)

    augmented = create_augmented_matrix(A, b)
    print_matrix(augmented, "Augmented Matrix [A | b]")

    upper_matrix = gaussian_elimination(A, b)

    print_matrix(
        upper_matrix,
        "Matrix After Forward Elimination"
    )

    solution = back_substitution(upper_matrix)

    print("\nSolution:")
    for i, value in enumerate(solution, start=1):
        print(f"x{i} = {value:.2f}")

    verified, calculated = verify_solution(A, b, solution)

    print("\nVerification:")
    print("Calculated A x x:", calculated)
    print("Original RHS b:  ", b)

    if verified:
        print("Result: Solution verified successfully.")
    else:
        print("Result: Verification failed.")

    return solution


# ---------------------------------------------------------
# TEST SYSTEM 1
# ---------------------------------------------------------

A1 = [
    [2, 1, -1],
    [-3, -1, 2],
    [-2, 1, 2]
]

b1 = [8, -11, -3]

print("=" * 60)
print("SYSTEM 1")
print("=" * 60)

try:
    solution1 = solve_system(A1, b1)
except ValueError as error:
    print("Error:", error)


# ---------------------------------------------------------
# TEST SYSTEM 2
# ---------------------------------------------------------

A2 = [
    [1, 2, 3],
    [2, -1, 1],
    [3, 1, -2]
]

b2 = [14, 3, 3]

print("\n" + "=" * 60)
print("SYSTEM 2")
print("=" * 60)

try:
    solution2 = solve_system(A2, b2)
except ValueError as error:
    print("Error:", error)

# ---------------------------------------------------------
# TEST SYSTEM 3
# ---------------------------------------------------------

A3 = [
    [2,1],
    [1,3]
]

b3 = [5,6]

print("\n" + "=" * 60)
print("SYSTEM 3")
print("=" * 60)

try:
    solution2 = solve_system(A3, b3)
except ValueError as error:
    print("Error:", error)
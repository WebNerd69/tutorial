# PROJECT REPORT 1

# Implementation of Gaussian Elimination Algorithm for Solving Linear Systems

## 1. Project Title

**Implementation of Gaussian Elimination Algorithm for Solving Linear Systems**

## 2. Objective

The objective of this project is to develop a generalized Python program for solving systems of linear equations using the **Gaussian Elimination algorithm**.

The program accepts a coefficient matrix and a right-hand-side (RHS) vector, forms the augmented matrix, converts it into upper triangular form using elementary row operations, and obtains the solution using back substitution.

The program also implements **partial pivoting** to handle zero or very small pivot elements and includes verification and error handling.

## 3. Introduction

A system of linear equations consists of two or more equations involving unknown variables. Such systems occur frequently in mathematics, engineering, computer science, statistics, and numerical computing.

A system of `n` linear equations can be represented in matrix form as:

\[
AX=B
\]

where:

- `A` is the coefficient matrix.
- `X` is the vector of unknown variables.
- `B` is the RHS vector.

Gaussian Elimination is a numerical method that transforms the system into an equivalent upper triangular system. The unknown variables can then be obtained through a process called **back substitution**.

In this project, the algorithm is implemented using Python functions so that it can be used for general square matrices rather than only for a specific example.

## 4. Mathematical Concept

Consider the following system:

\[
a_{11}x_1+a_{12}x_2+\cdots+a_{1n}x_n=b_1
\]

\[
a_{21}x_1+a_{22}x_2+\cdots+a_{2n}x_n=b_2
\]

\[
\vdots
\]

\[
a_{n1}x_1+a_{n2}x_2+\cdots+a_{nn}x_n=b_n
\]

This can be written as:

\[
AX=B
\]

For example:

\[
\begin{aligned}
2x+y-z&=8\\
-3x-y+2z&=-11\\
-2x+y+2z&=-3
\end{aligned}
\]

The coefficient matrix is:

\[
A=
\begin{bmatrix}
2&1&-1\\
-3&-1&2\\
-2&1&2
\end{bmatrix}
\]

and the RHS vector is:

\[
B=
\begin{bmatrix}
8\\
-11\\
-3
\end{bmatrix}
\]

## 5. Augmented Matrix

The coefficient matrix and RHS vector are combined to form the augmented matrix:

\[
[A|B]
\]

For the above system:

\[
\left[
\begin{array}{ccc|c}
2&1&-1&8\\
-3&-1&2&-11\\
-2&1&2&-3
\end{array}
\right]
\]

Gaussian Elimination applies elementary row operations to convert this matrix into upper triangular form.

## 6. Elementary Row Operations

The following row operations are used:

### Row interchange

\[
R_i \leftrightarrow R_j
\]

Two rows are exchanged.

### Row multiplication

\[
R_i \rightarrow kR_i
\]

A row is multiplied by a non-zero constant.

### Row replacement

\[
R_i \rightarrow R_i-kR_j
\]

A multiple of one row is subtracted from another row.

These operations preserve the solution of the system.

## 7. Partial Pivoting

Partial pivoting is used to handle zero or very small pivot elements.

At every elimination step, the program examines the values below the current pivot and selects the row containing the largest absolute value.

That row is then exchanged with the current row.

For example:

\[
\begin{bmatrix}
0&2&1\\
1&-1&2\\
2&1&-1
\end{bmatrix}
\]

The first pivot is zero. Therefore, the program searches the rows below it and swaps in a suitable row before continuing elimination.

This improves the numerical stability of the algorithm.

## 8. Forward Elimination

Forward elimination converts the augmented matrix into upper triangular form.

The general form after elimination is:

\[
\left[
\begin{array}{cccc|c}
u_{11}&u_{12}&u_{13}&\cdots&c_1\\
0&u_{22}&u_{23}&\cdots&c_2\\
0&0&u_{33}&\cdots&c_3\\
\vdots&\vdots&\vdots&\ddots&\vdots\\
0&0&0&\cdots&u_{nn}&c_n
\end{array}
\right]
\]

Once this form is obtained, the variables can be calculated starting from the last equation.

## 9. Back Substitution

Back substitution is implemented as a separate function.

For the last equation:

\[
u_{nn}x_n=c_n
\]

therefore:

\[
x_n=\frac{c_n}{u_{nn}}
\]

The value of \(x_n\) is then substituted into the previous equation to calculate \(x_{n-1}\).

The process continues until all unknowns are obtained.

The general formula is:

\[
x_i=
\frac{
c_i-\sum_{j=i+1}^{n}u_{ij}x_j
}{
u_{ii}
}
\]

## 10. Algorithm

1. Read the coefficient matrix `A`.
2. Read the RHS vector `B`.
3. Validate the dimensions of `A` and `B`.
4. Create the augmented matrix `[A|B]`.
5. Select the pivot element.
6. Apply partial pivoting.
7. Swap rows if necessary.
8. Eliminate elements below the pivot.
9. Repeat the process for all pivot positions.
10. Check for singular or inconsistent systems.
11. Pass the upper triangular matrix to the back-substitution function.
12. Calculate the unknown variables.
13. Substitute the solution into the original equations.
14. Verify the solution.

## 11. Program Structure

| Function | Purpose |
|---|---|
| `print_matrix()` | Displays matrices |
| `validate_input()` | Checks matrix dimensions |
| `create_augmented_matrix()` | Creates `[A|B]` |
| `gaussian_elimination()` | Performs elimination and pivoting |
| `back_substitution()` | Calculates the solution |
| `verify_solution()` | Verifies `AX=B` |
| `solve_system()` | Combines the complete process |

This modular approach makes the program reusable for different systems.

## 12. Test Case 1

The first system used for testing is:

\[
2x+y-z=8
\]

\[
-3x-y+2z=-11
\]

\[
-2x+y+2z=-3
\]

### Input Matrix

\[
A=
\begin{bmatrix}
2&1&-1\\
-3&-1&2\\
-2&1&2
\end{bmatrix}
\]

### RHS

\[
B=
\begin{bmatrix}
8\\
-11\\
-3
\end{bmatrix}
\]

### Obtained Solution

\[
\boxed{x=2,\quad y=3,\quad z=-1}
\]

### Verification

First equation:

\[
2(2)+3-(-1)=8
\]

Second equation:

\[
-3(2)-3+2(-1)=-11
\]

Third equation:

\[
-2(2)+3+2(-1)=-3
\]

Therefore:

\[
AX=B
\]

and the solution is verified successfully.

## 13. Test Case 2

The second system is:

\[
x+2y+3z=14
\]

\[
2x-y+z=3
\]

\[
3x+y-2z=-1
\]

The coefficient matrix is:

\[
A=
\begin{bmatrix}
1&2&3\\
2&-1&1\\
3&1&-2
\end{bmatrix}
\]

and:

\[
B=
\begin{bmatrix}
14\\
3\\
-1
\end{bmatrix}
\]

### Solution

\[
\boxed{x=1,\quad y=2,\quad z=3}
\]

### Verification

\[
1+2(2)+3(3)=14
\]

\[
2(1)-2+3=3
\]

\[
3(1)+2-2(3)=-1
\]

Therefore:

\[
\boxed{AX=B}
\]

and the solution is verified.

## 14. Additional Partial Pivoting Test

To demonstrate partial pivoting, the following system can be used:

\[
2y+z=4
\]

\[
x-y+2z=-1
\]

\[
2x+y-z=4
\]

with:

\[
A=
\begin{bmatrix}
0&2&1\\
1&-1&2\\
2&1&-1
\end{bmatrix}
\]

and:

\[
B=
\begin{bmatrix}
4\\
-1\\
4
\end{bmatrix}
\]

The solution is:

\[
\boxed{x=1,\quad y=2,\quad z=0}
\]

The first pivot is zero, so the program must exchange rows before performing elimination. This demonstrates the partial pivoting feature.

## 15. Error Handling

The program includes error handling for the following conditions:

### Invalid matrix dimensions

The program checks whether the coefficient matrix is square and whether the RHS vector has the correct number of elements.

### Zero pivot

A tolerance value is used to detect zero or extremely small pivot values.

### Singular system

If the elimination produces a row such as:

\[
[0\quad0\quad0|0]
\]

the system does not have a unique solution.

### Inconsistent system

If a row becomes:

\[
[0\quad0\quad0|c]
\]

where \(c\neq0\), the system has no solution.

## 16. Verification

The program verifies the calculated solution by computing:

\[
AX
\]

and comparing the result with the original RHS vector:

\[
B
\]

Because floating-point calculations can produce very small numerical errors, a tolerance is used:

\[
|Calculated-B|<10^{-8}
\]

If this condition is satisfied for every equation, the solution is considered verified.

## 17. Results and Observations

The implemented Gaussian Elimination algorithm successfully solves different systems of linear equations.

The program successfully performs:

- Augmented matrix formation.
- Partial pivoting.
- Forward elimination.
- Upper triangular matrix formation.
- Back substitution.
- Solution verification.
- Error detection.

The use of separate functions makes the implementation modular and allows the same program to solve systems of different sizes.

The partial pivoting test also demonstrates that the program can handle a zero pivot by exchanging rows.

## 18. Learning Outcomes

After completing this project, the following concepts were understood and implemented:

- Gaussian Elimination.
- Matrix representation of linear systems.
- Elementary row operations.
- Forward elimination.
- Partial pivoting.
- Back substitution.
- Numerical solution of linear systems.
- Verification of calculated solutions.
- Error handling.
- Generalized numerical programming.

## 19. Conclusion

The project successfully implements the Gaussian Elimination algorithm using Python.

The program is generalized and modular and can solve different square systems of linear equations. It forms an augmented matrix, performs forward elimination with partial pivoting, obtains the solution using back substitution, and verifies the result against the original system.

The project provides practical understanding of how a mathematical numerical algorithm can be converted into a reusable computational program.

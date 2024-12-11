#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>

#define MAX_DEGREE 100
#define EPSILON 1e-9
char flag = 0;

const char *dividend_str = "1000.00000001x^3 - 0.25x^2 + 3x + 4.5";
const char *divisor_str = "3x + 0.5";
// Example polynomials with very precise floating-point coefficients
// const char *dividend_str = "0.0000000x^3 + 0.000000x^2 + 0.000x + 1";
// const char *divisor_str = "0.5";

typedef struct
{
    int degree;
    long double coefficients[MAX_DEGREE + 1];
} Polynomial;

Polynomial parsePolynomial(const char *polyStr)
{
    Polynomial poly = {0};
    int strLen = strlen(polyStr);
    int coefficientSign = 1;
    long double coefficient = 0;
    long double fractionalPart = 0;
    long double fractionalMultiplier = 0.1;
    int currentDegree = 0;
    int readingCoeff = 1;
    int readingFraction = 0;
    int coeffSpecified = 0;
    int fractionDigits = 0;
    int highestNonZeroDegree = -1;

    for (int i = 0; i < strLen; i++)
    {
        if (polyStr[i] == ' ')
        {
            continue;
        }

        if (polyStr[i] == '+' || polyStr[i] == '-')
        {
            if ((coeffSpecified > 0 || currentDegree > 0) && (fabsl(coefficient) >= EPSILON || fabsl(fractionalPart) >= EPSILON))
            {
                long double finalCoeff = coefficientSign * (coefficient + fractionalPart);
                poly.coefficients[currentDegree] = finalCoeff;

                if (fabsl(finalCoeff) >= EPSILON)
                {
                    if (currentDegree > highestNonZeroDegree)
                    {
                        highestNonZeroDegree = currentDegree;
                    }
                }
            }

            if (polyStr[i] == '+')
            {
                coefficientSign = 1;
            }
            else
            {
                coefficientSign = -1;
            }

            coefficient = 0;
            fractionalPart = 0;
            fractionalMultiplier = 0.1;
            currentDegree = 0;
            readingCoeff = 1;
            readingFraction = 0;
            coeffSpecified = 0;
            fractionDigits = 0;
            continue;
        }

        if (isdigit(polyStr[i]))
        {
            coeffSpecified = 1;
            if (readingFraction)
            {
                if (fractionDigits < EPSILON - 1)
                {
                    fractionalPart += (polyStr[i] - '0') * fractionalMultiplier;
                    fractionalMultiplier *= 0.1;
                    fractionDigits++;
                }
            }
            else if (readingCoeff)
            {
                coefficient = coefficient * 10 + (polyStr[i] - '0');
            }
        }

        else if (polyStr[i] == '.')
        {
            readingFraction = 1;
            fractionalMultiplier = 0.1;
            fractionDigits = 0;
        }

        else if (polyStr[i] == 'x')
        {
            if (coeffSpecified == 0)
            {
                coefficient = 1;
                coeffSpecified = 1;
            }
            readingCoeff = 0;
            readingFraction = 0;

            // Check for degree
            if (i + 1 < strLen && polyStr[i + 1] == '^')
            {
                i += 2;
                currentDegree = 0;
                while (i < strLen && isdigit(polyStr[i]))
                {
                    currentDegree = currentDegree * 10 + (polyStr[i] - '0');
                    i++;
                }
                i--; // Compensate for outer loop increment
            }
            else
            {
                currentDegree = 1;
            }
        }
    }

    if ((coeffSpecified > 0 || currentDegree > 0) && (fabsl(coefficient) >= EPSILON || fabsl(fractionalPart) >= EPSILON))
    {
        long double finalCoeff = coefficientSign * (coefficient + fractionalPart);
        poly.coefficients[currentDegree] = finalCoeff;

        // Update highest non-zero degree
        if (fabsl(finalCoeff) >= EPSILON)
        {
            if (currentDegree > highestNonZeroDegree)
            {
                highestNonZeroDegree = currentDegree;
            }
        }
    }

    if (highestNonZeroDegree == -1)
    {
        poly.degree = 0;
    }
    else
    {
        poly.degree = highestNonZeroDegree;
    }
    return poly;
}

Polynomial multiplyPolynomials(Polynomial poly1, Polynomial poly2)
{
    Polynomial result = {0};

    for (int i = 0; i <= poly1.degree; i++)
    {
        for (int j = 0; j <= poly2.degree; j++)
        {
            result.coefficients[i + j] += poly1.coefficients[i] * poly2.coefficients[j];
        }
    }

    result.degree = poly1.degree + poly2.degree;
    while (fabsl(result.coefficients[result.degree]) < EPSILON && result.degree > 0)
    {
        result.degree--;
    }

    return result;
}

int printPolynomial(Polynomial poly)
{
    flag = 1;
    for (int i = poly.degree; i >= 0; i--)
    {
        if (fabsl(poly.coefficients[i]) < EPSILON)
        {
            continue;
        }

        printf("%+.8Lf", poly.coefficients[i]);
        if (i > 0)
        {
            printf("x");
            if (i > 1)
            {
                printf("^%d", i);
            }
        }

        flag = 0;
    }
    if (flag)
    {
        flag = 0;
        printf("0");
    }
    printf("\n");
    return 0;
}

int comparePolynomials(Polynomial poly1, Polynomial poly2)
{
    if (poly1.degree != poly2.degree)
    {
        return 0;
    }

    for (int i = 0; i <= poly1.degree; i++)
    {
        if (fabsl(poly1.coefficients[i] - poly2.coefficients[i]) > EPSILON)
        {
            return 0;
        }
    }

    return 1;
}

Polynomial polynomialDivision(Polynomial dividend, Polynomial divisor, Polynomial *remainder)
{
    Polynomial quotient = {0};

    while (dividend.degree >= divisor.degree)
    {
        if (flag == 1)
        {
            flag = 0;
            break;
        }
        if (divisor.degree == 0 && dividend.degree == 0)
        {
            flag = 1;
        }

        long double coefficient_ = dividend.coefficients[dividend.degree] / divisor.coefficients[divisor.degree];
        int degree_ = dividend.degree - divisor.degree;

        quotient.coefficients[degree_] = coefficient_;
        if (degree_ > quotient.degree)
        {
            quotient.degree = degree_;
        }

        for (int i = 0; i <= divisor.degree; i++)
        {
            dividend.coefficients[dividend.degree - i] -= coefficient_ * divisor.coefficients[divisor.degree - i];
        }

        while (dividend.degree > 0 && fabsl(dividend.coefficients[dividend.degree]) < EPSILON)
        {
            dividend.degree--;
        }
    }

    *remainder = dividend;

    return quotient;
}

int main()
{
    struct timespec start, end;
    long parsePolyTime, divisionPolyTime, multiplyPolyTime, comparePolyTime;
    clock_gettime(CLOCK_MONOTONIC, &start);

    Polynomial dividend = parsePolynomial(dividend_str);
    Polynomial divisor = parsePolynomial(divisor_str);
    Polynomial remainder;

    printf("Dividend: ");
    printPolynomial(dividend);
    printf("Divisor: ");
    printPolynomial(divisor);

    clock_gettime(CLOCK_MONOTONIC, &end);
    parsePolyTime = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);





    clock_gettime(CLOCK_MONOTONIC, &start);

    Polynomial quotient = polynomialDivision(dividend, divisor, &remainder);

    printf("Quotient: ");
    printPolynomial(quotient);
    printf("Remainder: ");
    printPolynomial(remainder);

    clock_gettime(CLOCK_MONOTONIC, &end);
    divisionPolyTime = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);






    clock_gettime(CLOCK_MONOTONIC, &start);

    Polynomial reconstructed = multiplyPolynomials(quotient, divisor);
    for (int i = 0; i <= remainder.degree; i++)
    {
        reconstructed.coefficients[i] += remainder.coefficients[i];
    }

    if (reconstructed.degree > remainder.degree)
    {
        reconstructed.degree = reconstructed.degree;
    }
    else
    {
        reconstructed.degree = remainder.degree;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    multiplyPolyTime = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);

    printf("\nReconstructed Dividend: ");
    printPolynomial(reconstructed);



    clock_gettime(CLOCK_MONOTONIC, &start);

    if (comparePolynomials(dividend, reconstructed))
    {
        printf("Verification: Polynomial division is CORRECT!\n\n");
    }
    else
    {
        printf("Verification: Polynomial division is INCORRECT!\n\n");
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    comparePolyTime = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);

    printf("Parsing dividend and divisor time: \t\t\t\t%ld ns\n", parsePolyTime);
    printf("Division dividend to divisor time: \t\t\t\t%ld ns\n", divisionPolyTime);
    printf("Multiplay quotient to divisor time: \t\t\t\t%ld ns\n", multiplyPolyTime);
    printf("Compare reconstructed polynomial with dividend time: \t\t%ld ns\n", comparePolyTime);

    return 0;
}
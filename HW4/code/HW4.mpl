
Rm := r;
R_1M := Rm -> Rm;
R_TMR := Rm -> Rm^3 + 3*Rm^2*(1 - Rm);
R_5MR := Rm -> Rm^5 + 5*Rm^4*(1 - Rm) + 10*Rm^3*(1 - Rm)^2;
R_7MR := Rm -> Rm^7 + 7*Rm^6*(1 - Rm) + 21*Rm^5*(1 - Rm)^2 + 35*Rm^4*(1 - Rm)^3;
plot([R_1M(r), R_TMR(r), R_5MR(r), R_7MR(r)], r = 0 .. 1, labels = ["Reliability of Single Module (r)", "System Reliability"], legend = ["1M", "TMR", "5MR", "7MR"], title = "Comparison of 1M, TMR, 5MR, and 7MR System Reliability");
plot_log_reliability := proc() local log_TMR, log_5MR, log_7MR, log_1M; log_1M := r -> -log10(1 - R_1M(r)); log_TMR := r -> -log10(1 - R_TMR(r)); log_5MR := r -> -log10(1 - R_5MR(r)); log_7MR := r -> -log10(1 - R_7MR(r)); plot([log_1M(r), log_TMR(r), log_5MR(r), log_7MR(r)], r = 0.99 .. 1, labels = ["Reliability of Single Module (r)", "Logarithmic Reliability Measure"], legend = ["1M", "TMR", "5MR", "7MR"], thickness = [4, 5, 6, 7], linestyle = [dot, dot, dot, dot], gridlines = true, title = "Logarithmic Reliability for 1M, TMR, 5MR, and 7MR"); end proc;
plot_log_reliability();
r_values := [0.999, 0.99999, 0.9999999];
reliability_functions := [R_1M, R_TMR, R_5MR, R_7MR];
function_names := ["R_1M", "R_TMR", "R_5MR", "R_7MR"];
printf("Reliability Values for Different Systems:\n\n");
printf("\t\t\t\tr=0.999\t\t\t\t\tr=0.99999\t\t\tr=0.9999999\n");
printf("---------------------------------------------------------------------------------------------------------------------\n");
for i to 4 do
    printf("%s\t", function_names[i]);
    for r in r_values do
        printf("\t|\t\t%.10f", evalf(reliability_functions[i](r)));
    end do;
    printf("\n");
end do;
printf("---------------------------------------------------------------------------------------------------------------------\n");


lambda := 50;
C_values := [seq(i, i = 0 .. 1, 0.1)];
NULL;
color_list := [cyan, yellow, purple, pink, brown, blue, orange, red, black, magenta, green];
R := Tsys -> (C, t) -> C*(2*exp(-lambda*t) - exp(-2*lambda*t)) + (1 - C)*exp(-lambda*t);
combined_plot := plot([seq(R(Tsys)(C, t), C = C_values)], t = 0 .. 0.02, title = "Plot of R_Tsys for Different Values of C", labels = ["t", "R_Tsys"], color = color_list, thickness = 2);


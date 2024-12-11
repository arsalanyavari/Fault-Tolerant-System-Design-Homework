
# 
# 
SeriesReliability := proc(blocks::list) local i, R; R := 1; for i in blocks do R := R*i; end do; return R; end proc;
ParallelReliability := proc(blocks::list) local i, R; R := 1; for i in blocks do R := R*(1 - i); end do; return 1 - R; end proc;
ParseRBD := proc(rbdString::string) local parsedString; parsedString := StringTools:-SubstituteAll(rbdString, "S(", "SeriesReliability(["); parsedString := StringTools:-SubstituteAll(parsedString, "P(", "ParallelReliability(["); parsedString := StringTools:-SubstituteAll(parsedString, ")", "])"); return eval(parse(parsedString)); end proc;
RemoveB := proc(rbdString::string) local openBracketPos, closeBracketPos, cleanedString; cleanedString := rbdString; while StringTools:-Search("B(", cleanedString) <> 0 do openBracketPos := StringTools:-Search("B(", cleanedString); closeBracketPos := StringTools:-Search(")", cleanedString, openBracketPos); cleanedString := StringTools:-Substitute(cleanedString, cleanedString[openBracketPos .. openBracketPos + 1], ""); cleanedString := StringTools:-Substitute(cleanedString, cleanedString[closeBracketPos - 2 .. closeBracketPos - 2], ""); end do; return cleanedString; end proc;
ReadRBDFromFile := proc(filename::string) local rbdString; rbdString := FileTools:-Text:-ReadFile(filename); rbdString := StringTools:-Trim(rbdString); rbdString := RemoveB(rbdString); return rbdString; end proc;
ComputeParametricReliabilityFromFile := proc(filename::string) local rbdString, reliability; rbdString := ReadRBDFromFile(filename); print("RBD String from file: ", rbdString); reliability := ParseRBD(rbdString); return reliability; end proc;
filename := "/home/andre/code/uni/fault/midterm/Q3/parametric2_rbd.txt";
reliability := ComputeParametricReliabilityFromFile(filename);
print("The symbolic reliability of the system is: ", reliability);
NULL;


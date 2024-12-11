import sys
import re
import subprocess
# flag = 0

def process_cpp(input_file):
    with open(input_file, 'r') as file:
        content = file.read()

    pattern = r'(\b\w+\s*\(.*?\))\s*\n\s*\{'
    replacement = r'\1{'
    new_content = re.sub(pattern, replacement, content)

    with open(input_file, 'w') as file:
        file.write(new_content)

def makes_fault_tolerant(input_file, output_file):
    flag = 0
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        lines = infile.readlines()
        id_counter = 0
        stack_name = 'id_stack'

        outfile.write(f"#include <stack>\nstd::stack<int> {stack_name};\n\n")

        inside_function = False
        braces_count = 0
        inside_multiline_comment = False

        control_keywords = ['if', 'for', 'while', 'switch']

        for line in lines:
            # Check for start of multi-line comment
            if '/*' in line and not inside_multiline_comment:
                inside_multiline_comment = True
                outfile.write(line)
                continue

            # Check for end of multi-line comment
            if '*/' in line and inside_multiline_comment:
                inside_multiline_comment = False
                outfile.write(line)
                continue

            # Skip entire line if inside multi-line comment
            if inside_multiline_comment:
                outfile.write(line)
                continue

            stripped_line = line.strip()

            # Skip single-line comments
            if re.match(r'//.*', stripped_line):
                outfile.write(line)
                continue

            # Check if the line starts with any control keywords
            if any(re.match(rf'^\s*{keyword}\b', stripped_line) for keyword in control_keywords):
                outfile.write(line)
                braces_count += stripped_line.count('{')
                braces_count -= stripped_line.count('}')
                continue
            
            if re.match(r'.*?\(\s*?.*?\s*?\)\s*?.*?{\s*?}.*?', stripped_line):
                last_index = len(stripped_line) - 1 - stripped_line[::-1].index('}')
                outfile.write(stripped_line[:last_index])
                outfile.write(f"\n    {stack_name}.push({id_counter});\n")
                id_counter += 1
                outfile.write(f"    if ({stack_name}.top() != {id_counter - 1}) {{\n")
                outfile.write(f'        std::cerr << "Stack mismatch error!\\n";\n')
                outfile.write(f"    }}\n")
                outfile.write(f"    {stack_name}.pop();\n")
                outfile.write(stripped_line[last_index:])
                continue

            if re.match(r'^(?!.*=).*?\(.*\)\s*?.*?{', stripped_line):
                braces_count = 0
                outfile.write(line)
                inside_function = True
                outfile.write(f"    {stack_name}.push({id_counter});\n")
                id_counter += 1
                braces_count += stripped_line.count('{')
                continue

            if inside_function:
                braces_count += stripped_line.count('{')
                braces_count -= stripped_line.count('}')

                if 'return' in stripped_line:
                    outfile.write(f"    if ({stack_name}.top() != {id_counter - 1}) {{\n")
                    outfile.write(f'        std::cerr << "Stack mismatch error!\\n";\n')
                    outfile.write(f"    }}\n")
                    outfile.write(f"    {stack_name}.pop();\n")
                    flag = 1
                    # inside_function = False

                if braces_count == 0:
                    if (flag == 1):
                        flag = 0
                        inside_function = False
                    else:
                        inside_function = False
                        outfile.write(f"    if ({stack_name}.top() != {id_counter - 1}) {{\n")
                        outfile.write(f'        std::cerr << "Stack mismatch error!\\n";\n')
                        outfile.write(f"    }}\n")
                        outfile.write(f"    {stack_name}.pop();\n")

            outfile.write(line)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python Q2.py sample.cpp out.cpp")
        sys.exit(1)

    input_cpp_file = sys.argv[1]
    output_cpp_file = sys.argv[2]
    subprocess.run(['clang-format', '-i', input_cpp_file])

    process_cpp(input_cpp_file)
    makes_fault_tolerant(input_cpp_file, output_cpp_file)

    # subprocess.run(['clang-format', '-i', "Q2/sample.cpp"])
    # process_cpp("Q2/sample.cpp")
    # makes_fault_tolerant("Q2/sample.cpp", "Q2/out.cpp")

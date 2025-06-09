# %%
def truncate_lines_in_file(filename, max_length=150, truncation_marker='...'):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    truncated_lines = []
    for line in lines:
        line = line.rstrip('\n')  # Remove newline for accurate length check
        if len(line) > max_length:
            truncated_line = line[:max_length - len(truncation_marker)] + truncation_marker
        else:
            truncated_line = line
        truncated_lines.append(truncated_line + '\n')

    with open(filename, 'w', encoding='utf-8') as file:
        file.writelines(truncated_lines)

truncate_lines_in_file(r'C:\Users\kevin\dev\tornado-tree-destruction-ef\resources\annotations copy.xml')




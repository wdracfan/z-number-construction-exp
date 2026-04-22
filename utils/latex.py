def table_to_latex(table: list[list[str]], bold_max_columns: list[int], max_or_min: str = 'max') -> str:
    for col in bold_max_columns:
        max_row = None
        max_value = -1e9 if max_or_min == 'max' else 1e9
        for row in range(len(table)):
            if (max_or_min == 'max' and float(table[row][col]) > max_value) or (max_or_min == 'min' and float(table[row][col]) < max_value):
                max_value = float(table[row][col])
                max_row = row
        table[max_row][col] = f'\\textbf{{{table[max_row][col]}}}'
    result = ' \\\\ \\hline\n'.join([' & '.join(row) for row in table])
    return result
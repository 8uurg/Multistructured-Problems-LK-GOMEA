def get_sat_from_file(file): 
    line = file.readline() 
    while line.startswith('c'): line = file.readline() 
    header = line.split() 
    assert header[0] == 'p' 
    assert header[1] == 'cnf' 
    L = int(header[2]) 
    C = int(header[3]) 
    clauses = [] 
    for _ in range(C): 
        line = file.readline() 
        while line.startswith('c'): line = file.readline() 
        clauses.append(tuple([int(a) for a in line.split()][:-1])) 
    return L, C, clauses

def get_clause_satisfier(clause, variables): 
    for v in clause: 
        if variables[abs(v) - 1] == ('1' if v > 0 else '0'): 
            # print(f"Clause {clause} is satisfied by variable {abs(v)} ({variables[abs(v) - 1]})") 
            return v 
    return 0

def is_clause_satisfied(clause, variables): 
    return get_clause_satisfier(clause, variables) != 0

def sat_model_to_string(m):
    return ''.join('0' if s.startswith("-") else '1' for s in m.split())
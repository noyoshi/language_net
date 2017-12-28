#!/usr/bin/env python3.5 

import sys 
import subprocess 

def temp_parse(line):
    # I messed up grep/diff when I made the files, so I am fixing that specfic
    # error with this  
    line = line.strip("<")
    line = line.strip()
    
    if line[0].isdigit(): 
        return None 
    
    return line + "\n" 





if __name__ == '__main__':
    args = sys.argv[1:]

    for filename in args: 
        filename = str(filename) 
        with open(filename, "r") as _file:
            contents = _file.readlines()
            updated_contents = []
            for line in contents: 
                sanitized_line = temp_parse(line)
                if sanitized_line: 
                    updated_contents.append(sanitized_line)

        # Renames the old, unsanitized list to "old_filename"
        subprocess.run(["mv", filename, "old_" + filename])
        
        # Writes the sanitized words to the file 
        with open(filename, "w") as save:
            for line in updated_contents: 
                save.write(line)


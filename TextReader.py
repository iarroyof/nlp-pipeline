class TextReader:
    """Print and number lines in a text file."""

    def __init__(self, filename, N = 1):
        self.filename = filename
        self.file = open(filename)
        self.lineno = 0
        self.N = N

    def read(self):
        from numpy import array, zeros
        
        self.lineno += 1
        if self.N > 1:
            
            for n in xrange(self.N):
                line = self.file.readline()
                if not line:
                    continue
                if line.endswith('\n'):
                    line = line[:-1]
                    
            return array(map(float, line.split()))
        else:
            with open(self.file) as f:
                for line in f:
                    if not line:
                        continue
                    if line.endswith('\n'):
                        line = line[:-1]
                    return array(map(float, line.split()))    

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['file']
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., filename and lineno).
        self.__dict__.update(state)
        # Restore the previously opened file's state. To do so, we need to
        # reopen it and read from it until the line count is restored.
        file = open(self.filename)
        for _ in range(self.lineno):
            file.readline()
        # Finally, save the file.    self.file = file

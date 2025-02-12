from abc import ABC, abstractmethod


class StructureIterator(ABC):
    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass


class FileStructureIterator(StructureIterator):
    def __init__(self, file_path):
        self.file = open(file_path)
        self.buffer = []
        self.iterator = iter(self.file)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self.buffer:
                return self.buffer.pop(0)
            try:
                line = next(self.iterator)
                self.buffer = line.strip().split()
            except StopIteration:
                self.close()
                raise

    def close(self):
        if self.file and not self.file.closed:
            self.file.close()

    def __del__(self):
        self.close()


class ListStructureIterator(StructureIterator):
    def __init__(self, structure_list):
        self.iterator = iter(structure_list)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            line = next(self.iterator)
            return line
        except StopIteration:
            raise

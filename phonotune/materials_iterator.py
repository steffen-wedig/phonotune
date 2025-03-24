from abc import ABC, abstractmethod

# Classes to iterate over collections of mp-id.


class MaterialsIterator(ABC):
    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass


class FileMaterialsIterator(MaterialsIterator):
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


class ListMaterialsIterator(MaterialsIterator):
    def __init__(self, materials_list):
        self.iterator = iter(materials_list)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            line = next(self.iterator)
            return line
        except StopIteration:
            raise

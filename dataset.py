class Dataset():
    # dict id(int) -> PhoneNumbers

    def __init__(self):
        self.nums = dict()

    def __getitem__(self, key):
        return self.nums[key]

    def __setitem__(self, key, val):
        self.nums[key] = val

    def __str__(self):
        return str(self.nums)

    def __len__(self):
        return len(self.nums)

    def contains(self, item):
        return item in self.nums

    def values(self):
        return self.nums.values()
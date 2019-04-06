class Dataset():
    # a list of PhoneNumbers

    def __init__(self):
        self.nums = []
    
    def __getitem__(self, idx):
        return self.nums[idx]
    
    def __setitem__(self, idx, val):
        self.nums[idx] = val
    
    def __str__(self):
        return str(self.nums)
    
    def __len__(self):
        return len(self.nums)
    
    def append(self, item):
        self.nums.append(item)
    
    def extend(self, other_list):
        self.nums.extend(other_list)
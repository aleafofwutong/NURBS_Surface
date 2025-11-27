import torch

class HashGrid:
    def __init__(self, bounding_box, resolution=128, hash_size=2**20):
        """
        初始化哈希网格
        :param bounding_box: 空间边界 [x_min, x_max, y_min, y_max, z_min, z_max]
        :param resolution: 网格分辨率
        :param hash_size: 哈希表大小
        """
        self.bbox = torch.tensor(bounding_box, device='cuda', dtype=torch.float32)
        self.res = resolution
        self.hash_size = hash_size
        self.cell_size = (self.bbox[1::2] - self.bbox[::2]) / resolution
        
        # 哈希表：存储每个网格单元包含的点索引
        self.hash_table = [[] for _ in range(hash_size)]
        
        # 随机哈希函数参数
        self.hash_params = torch.randint(0, 2**20, (3, hash_size), device='cuda', dtype=torch.int32)

    def _hash(self, indices):
        """计算三维网格索引的哈希值"""
        x, y, z = indices.unbind(1)
        return (x * self.hash_params[0] ^ y * self.hash_params[1] ^ z * self.hash_params[2]) % self.hash_size

    def build(self, points):
        """构建哈希网格"""
        # 计算每个点所在的网格索引
        indices = ((points - self.bbox[::2].unsqueeze(0)) / self.cell_size.unsqueeze(0)).floor().long()
        
        # 过滤超出边界的点
        mask = torch.all((indices >= 0) & (indices < self.res), dim=1)
        valid_indices = indices[mask]
        valid_points = points[mask]
        
        # 计算哈希值并填充哈希表
        hashes = self._hash(valid_indices)
        for i in range(valid_points.shape[0]):
            self.hash_table[hashes[i]].append(i)
            
        return valid_points, valid_indices

    def query(self, query_points, radius):
        """查询指定半径内的点"""
        # 计算查询点所在网格及相邻网格
        cell_radius = (radius / self.cell_size).ceil().int()
        indices = ((query_points - self.bbox[::2].unsqueeze(0)) / self.cell_size.unsqueeze(0)).floor().long()
        
        neighbors = []
        for i in range(query_points.shape[0]):
            x, y, z = indices[i]
            # 检查周围网格
            for dx in range(-cell_radius[0], cell_radius[0]+1):
                for dy in range(-cell_radius[1], cell_radius[1]+1):
                    for dz in range(-cell_radius[2], cell_radius[2]+1):
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if 0 <= nx < self.res and 0 <= ny < self.res and 0 <= nz < self.res:
                            h = self._hash(torch.tensor([[nx, ny, nz]], device='cuda'))[0]
                            neighbors.extend(self.hash_table[h])
        return torch.unique(torch.tensor(neighbors, device='cuda'))
if __name__ == "__main__":
    # 示例用法
    bounding_box = [0, 1, 0, 1, 0, 1]
    grid = HashGrid(bounding_box, resolution=64, hash_size=2**16)
    
    # 随机生成点云
    points = torch.rand((1000, 3), device='cuda')
    
    # 构建哈希网格
    valid_points, valid_indices = grid.build(points)
    
    # 查询点
    query_points = torch.tensor([[0.5, 0.5, 0.5]], device='cuda')
    radius = 0.1
    
    neighbors = grid.query(query_points, radius)
    print("Neighbors indices:", neighbors)
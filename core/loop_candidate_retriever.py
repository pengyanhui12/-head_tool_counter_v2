"""回环候选检索器——MVP 间隔采样"""
class LoopCandidateRetriever:
    def __init__(self, sample_interval: int = 10, exclude_recent: int = 10):
        self.sample_interval = sample_interval
        self.exclude_recent = exclude_recent

    def get_candidates(self, current_node_id: int, num_nodes: int) -> list[int]:
        upper = max(0, current_node_id - self.exclude_recent)
        return list(range(0, upper, self.sample_interval))

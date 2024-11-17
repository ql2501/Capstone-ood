# main model code for negprompt
from dassl.engine import TRAINER_REGISTRY, TrainerX


@TRAINER_REGISTRY.register()
class NegPrompt(TrainerX):
    """
        Dummy class for NegPrompt for config debugging
    """

    # dummy method 
    def load_model(self, directory, epoch=None): 
        print("Calling CoOp_works\\CoOp\\trainers\\negprompt.NegPrompt.load_model")

    # dummy method 
    def test(self): 
        print("Calling CoOp_works\\CoOp\\trainers\\negprompt.NegPrompt.test")

    # dummy method
    # CoOp_works\Dassl.pytorch\dassl\engine\trainer.py 第324行，会在initialize SimpleTrainer的时候call build_model
    # 所以这里要override一下
    def build_model(self):
        print("Calling CoOp_works\\CoOp\\trainers\\negprompt.NegPrompt.build_model")

    # 之后train应该会用到
    def forward_backward(self, batch):
        raise NotImplementedError
    
    # 之后train应该会用到
    def parse_batch_train(self, batch):
        raise NotImplementedError
    
    # override SimpleTrainer的train()
    def train(self): 
        print("Calling CoOp_works\\CoOp\\trainers\\negprompt.NegPrompt.train")
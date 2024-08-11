from clearml import Task 
import torch
from pathlib import Path


class Policy(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.network(state)



def main():
    task = Task.init(project_name="RL Implementations", task_name="test task") 


    number_of_samples = 100

    input_dim = 2
    hidden_dim = 32
    output_dim = 2
    epochs = 10000

    policy = Policy(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    state = torch.randn(number_of_samples, input_dim)
    state = state + state.min(dim=1, keepdim=True)[0].abs()
    state = state/torch.sum(state, dim=1, keepdim=True)

    output = state

    for epoch in range(epochs):
        optimizer.zero_grad()
        action_probabilities = policy(state)
        loss = torch.nn.functional.mse_loss(action_probabilities, output)
        if epoch % 100 == 0:
            print("Epoch: ", epoch, "Loss: ", loss.item())
        loss.backward()
        optimizer.step()

    # Create directory if it doesn't exist
    model_dir = Path("modelRegistry")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Create the file if it doesn't exist
    model_path = model_dir / "vanilla-policy-gradient.pt"
    model_path.touch(exist_ok=True)

    torch.save(policy.state_dict(), model_path)
    task.upload_artifact(name="vanilla-policy-gradient", artifact_object=model_path)

    print("Training Done", loss.item(), output, action_probabilities)


if __name__ == '__main__':
    main()
import torch


def is_ubuntu() -> bool:
	try:
		with open("/etc/os-release") as f:
			for line in f:
				if line.startswith("NAME") and "Ubuntu" in line:
					return True
	except FileNotFoundError:
		pass
	return False


def is_github_action() -> bool:
	return is_ubuntu() and not torch.cuda.is_available()

variable "aws_region" {
  description = "AWS region where the benchmark instance will be created."
  type        = string
  default     = "us-east-1"
}

variable "name_prefix" {
  description = "Prefix used for AWS resource names and tags."
  type        = string
  default     = "nlp-benchmark"
}

variable "service_type" {
  description = "Which service to run on EC2: traditional or ollama."
  type        = string
  default     = "traditional"

  validation {
    condition     = contains(["traditional", "ollama"], var.service_type)
    error_message = "service_type must be either traditional or ollama."
  }
}

variable "instance_type" {
  description = "EC2 instance type. Examples: t2.nano, t2.micro, c7a.large, c7a.xlarge, c7a.2xlarge."
  type        = string
  default     = "t2.nano"
}

variable "availability_zone" {
  description = "Optional availability zone override. Leave blank to choose a compatible default subnet automatically."
  type        = string
  default     = ""
}

variable "repo_url" {
  description = "Git URL that EC2 can clone. HTTPS is easiest for public repos."
  type        = string
  default     = "https://github.com/<owner>/<repo>.git"
}

variable "repo_ref" {
  description = "Branch, tag, or commit to check out after cloning the repo."
  type        = string
  default     = "master"
}

variable "service_port" {
  description = "Port exposed by the sentiment service."
  type        = number
  default     = 8000
}

variable "allowed_app_cidr" {
  description = "CIDR allowed to call the service port. Leave blank to auto-detect your current public IP."
  type        = string
  default     = ""
}

variable "allowed_ssh_cidr" {
  description = "CIDR allowed to SSH. Leave blank to auto-detect your current public IP."
  type        = string
  default     = ""
}

variable "key_name" {
  description = "Optional EC2 key pair name for SSH access. Leave empty if you do not need SSH."
  type        = string
  default     = ""
}

variable "ssh_private_key_path" {
  description = "Optional local private key path used only to print a convenient ssh command."
  type        = string
  default     = ""
}

variable "root_volume_size_gb" {
  description = "Root disk size. Ollama needs extra space for model downloads."
  type        = number
  default     = 30
}

variable "ollama_model" {
  description = "Ollama model to pull and use when service_type is ollama."
  type        = string
  default     = "llama3.2:3b-instruct-q3_K_S"
}

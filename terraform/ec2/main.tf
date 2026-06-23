terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    http = {
      source  = "hashicorp/http"
      version = "~> 3.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

locals {
  repo_dir = "/home/ec2-user/Special-Problems"

  service_name = var.service_type == "ollama" ? "ollama-sentiment" : "traditional-sentiment"

  current_public_ip_cidr = "${trimspace(data.http.current_public_ip.response_body)}/32"
  app_cidr               = var.allowed_app_cidr == "" ? local.current_public_ip_cidr : var.allowed_app_cidr
  ssh_cidr               = var.allowed_ssh_cidr == "" ? local.current_public_ip_cidr : var.allowed_ssh_cidr
  subnet_id              = sort(data.aws_subnets.compatible.ids)[0]

  setup_command = var.service_type == "ollama" ? (
    "bash services/setup_ec2_ollama_service.sh --repo-dir '${local.repo_dir}' --port ${var.service_port} --service-user ec2-user --model '${var.ollama_model}'"
    ) : (
    "bash services/setup_ec2_service.sh --repo-dir '${local.repo_dir}' --port ${var.service_port} --service-user ec2-user"
  )
}

data "http" "current_public_ip" {
  url = "https://checkip.amazonaws.com/"
}

data "aws_vpc" "default" {
  default = true
}

data "aws_ec2_instance_type_offerings" "compatible" {
  location_type = "availability-zone"

  filter {
    name   = "instance-type"
    values = [var.instance_type]
  }
}

data "aws_subnets" "compatible" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }

  filter {
    name = "availability-zone"
    values = var.availability_zone == "" ? (
      data.aws_ec2_instance_type_offerings.compatible.locations
      ) : (
      [var.availability_zone]
    )
  }
}

data "aws_subnet" "selected" {
  id = local.subnet_id
}

data "aws_ami" "amazon_linux_2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-2023.*-x86_64"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_security_group" "sentiment_service" {
  name_prefix = "${var.name_prefix}-${var.service_type}-"
  description = "Access for the sentiment benchmark service"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "Service HTTP port"
    from_port   = var.service_port
    to_port     = var.service_port
    protocol    = "tcp"
    cidr_blocks = [local.app_cidr]
  }

  ingress {
    description = "SSH for setup/debugging"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [local.ssh_cidr]
  }

  egress {
    description = "Outbound internet access for package/model downloads"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "${var.name_prefix}-${var.service_type}-sg"
    Project = var.name_prefix
  }
}

resource "aws_instance" "sentiment_service" {
  ami                         = data.aws_ami.amazon_linux_2023.id
  instance_type               = var.instance_type
  subnet_id                   = local.subnet_id
  associate_public_ip_address = true
  key_name                    = var.key_name == "" ? null : var.key_name
  vpc_security_group_ids      = [aws_security_group.sentiment_service.id]

  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    repo_url      = var.repo_url
    repo_ref      = var.repo_ref
    repo_dir      = local.repo_dir
    service_type  = var.service_type
    service_name  = local.service_name
    setup_command = local.setup_command
    service_port  = var.service_port
  })

  user_data_replace_on_change = true

  root_block_device {
    volume_size = var.root_volume_size_gb
    volume_type = "gp3"
  }

  tags = {
    Name        = "${var.name_prefix}-${var.service_type}-${var.instance_type}"
    Project     = var.name_prefix
    ServiceType = var.service_type
  }
}

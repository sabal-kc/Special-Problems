output "instance_id" {
  description = "EC2 instance id."
  value       = aws_instance.sentiment_service.id
}

output "public_ip" {
  description = "Public IP address of the benchmark service instance."
  value       = aws_instance.sentiment_service.public_ip
}

output "health_url" {
  description = "Health check URL."
  value       = "http://${aws_instance.sentiment_service.public_ip}:${var.service_port}/health"
}

output "endpoint_url" {
  description = "Sentiment benchmark endpoint to put in benchmarks/task1_config.yaml."
  value       = "http://${aws_instance.sentiment_service.public_ip}:${var.service_port}/v1/sentiment"
}

output "service_name" {
  description = "Systemd service name on the EC2 instance."
  value       = local.service_name
}

output "allowed_app_cidr" {
  description = "CIDR allowed to call the service port."
  value       = local.app_cidr
}

output "allowed_ssh_cidr" {
  description = "CIDR allowed to SSH."
  value       = local.ssh_cidr
}

output "ssh_url" {
  description = "SSH URL for the EC2 instance."
  value       = "ssh://ec2-user@${aws_instance.sentiment_service.public_ip}"
}

output "ssh_command" {
  description = "Copy-paste SSH command. Set ssh_private_key_path to include -i automatically."
  value = var.ssh_private_key_path == "" ? (
    "ssh ec2-user@${aws_instance.sentiment_service.public_ip}"
    ) : (
    "ssh -i ${var.ssh_private_key_path} ec2-user@${aws_instance.sentiment_service.public_ip}"
  )
}

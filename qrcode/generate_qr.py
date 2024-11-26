import qrcode

# GitHub repository URL
repo_url = "https://github.com/NS027/CS7980_Capstone_RBCMuseum"

# Generate QR code
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)
qr.add_data(repo_url)
qr.make(fit=True)

# Save the QR code as an image
img = qr.make_image(fill_color="black", back_color="white")
img.save("github_repo_qr.png")

print("QR code saved as github_repo_qr.png")
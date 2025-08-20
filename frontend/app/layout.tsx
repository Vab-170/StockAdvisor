import './globals.css'

export const metadata = {
	title: 'Stock Advisor',
	description: 'Real-time stock market data and analysis',
}

export default function RootLayout({
	children,
}: {
	children: React.ReactNode
}) {
	return (
		<html lang="en">
			<body>
				<main>{children}</main>
			</body>
		</html>
	)
}

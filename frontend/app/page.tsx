'use client';

import { useState, useEffect } from 'react';

export default function Home() {
	const [data, setData] = useState(null);
	const [loading, setLoading] = useState(true);

	useEffect(() => {
		const fetchData = async () => {
			try {
				const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/stocks`);
				const result = await response.json();
				setData(result);
			} catch (error) {
				console.error('Error:', error);
			} finally {
				setLoading(false);
			}
		};

		fetchData();
	}, []);

	return (
		<div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
			<h1>Stock Advisor</h1>

			{loading ? (
				<p>Loading...</p>
			) : (
				<div>
					<h2>API Response:</h2>
					<pre style={{
						background: '#f5f5f5',
						padding: '10px',
						borderRadius: '5px',
						overflow: 'auto'
					}}>
						{JSON.stringify(data, null, 2)}
					</pre>
				</div>
			)}
		</div>
	);
}

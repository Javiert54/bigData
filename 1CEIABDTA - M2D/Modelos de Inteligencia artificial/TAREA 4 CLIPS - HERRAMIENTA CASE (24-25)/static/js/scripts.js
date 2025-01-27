document.getElementById('generate-diagram').addEventListener('click', 
	async () => { 
		try { 
			// Realizar la solicitud POST al servidor Flask 
			const response = await fetch('/process-diagram', { 
				method: 'POST', 
				headers: { 
					'Content-Type': 'application/json', 
				}, 
			}); 
	 
			// Manejar la respuesta 
			const result = await response.json(); 
			const statusDiv = document.getElementById('status'); 
	 
			if (response.ok) { 
				statusDiv.innerHTML = `<p style="color: green;">${result.message}</p>`; 
			} else { 
				statusDiv.innerHTML = `<p style="color: red;">Error: ${result.error}</p>`; 
			} 
		} catch (error) { 
			console.error('Error al procesar el diagrama:', error); 
	        document.getElementById('status').innerHTML = `<p style="color: red;">Error inesperado: ${error.message}</p>`; 
    } 
})
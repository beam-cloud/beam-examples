import { useEffect, useState } from "react";
import beam from "@beamcloud/beam-js";

function App() {
  const [deployment, setDeployment] = useState([]);
  const [apiResponse, setApiResponse] = useState("");

  const getDeployment = async () => {
    const client = await beam.init(process.env.REACT_APP_BEAM_TOKEN);
    const dep = await client.deployments.list();
    setDeployment(dep);
  };

  useEffect(() => {
    getDeployment();
  }, []);

  return (
    <div className="App">
      <table>
        <thead>
          <td>Name</td>
          <td>Id</td>
          <td>Type</td>
          <td>Make API Request</td>
        </thead>
        <tbody>
          {deployment.map((dep) => {
            return (
              <tr>
                <td>{dep.data.name}</td>
                <td>{dep.data.id}</td>
                <td>{dep.data.stub_type}</td>
                <td>
                  <button
                    onClick={async () => {
                      const resp = await dep.call();
                      setApiResponse(resp.data);
                    }}
                  >
                    Make API Request
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div>
        <h2>API Response</h2>
        <pre>{JSON.stringify(apiResponse, null, 2)}</pre>
      </div>
    </div>
  );
}

export default App;
